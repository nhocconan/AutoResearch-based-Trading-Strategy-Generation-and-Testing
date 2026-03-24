#!/usr/bin/env python3
"""
Experiment #387: 6h Primary + 1d HTF — Simplified Regime Pullback v1

Hypothesis: Previous 6h strategies failed due to overly complex regime detection
(ADX+CHOP combos) and too many confluence requirements. This version uses:
1. SIMPLE regime: ADX > 25 = trend, ADX < 20 = chop (no CHOP filter)
2. TREND entries: pullback to HMA(21) in direction of 1d HMA trend
3. CHOP entries: RSI(7) extremes with wider bands (15/85 instead of 25/75)
4. MAX 3 conditions per entry to ensure trades actually trigger
5. Position tracking with proper stoploss at 2.5x ATR

Why 6h might work: Between 4h (too many trades) and 12h (too few), 6h captures
multi-day swings while keeping trade count manageable (target 30-50/year).

Key differences from failed #380, #383:
- Removed CHOP index (unreliable on 6h)
- RSI(7) instead of RSI(14) for faster signals
- Looser RSI thresholds (15/85) for more trades
- Single HTF (1d) not triple (1d+1w+12h)
- Simpler entry: just HMA pullback + HTF alignment

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_simple_regime_pullback_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    hma_6h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    rsi_fast = calculate_rsi(close, period=7)  # Faster RSI for 6h
    rsi_slow = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi_fast[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SIMPLE REGIME DETECTION ===
        # Trending: ADX > 25
        # Choppy: ADX < 20
        # Otherwise: maintain previous state
        
        is_trending = adx[i] > 25.0
        is_choppy = adx[i] < 20.0
        
        # === HTF BIAS (1d) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === HMA PULLBACK DETECTION ===
        # Price pulled back to HMA but trend still intact
        pullback_long = False
        pullback_short = False
        
        if i > 1:
            # Long pullback: was above HMA, now near/touching HMA, still above SMA200
            if hma_bull and htf_1d_bull:
                price_above_hma_prev = close[i-1] > hma_6h[i-1] if not np.isnan(hma_6h[i-1]) else False
                price_near_hma = abs(close[i] - hma_6h[i]) < 0.015 * close[i]  # Within 1.5%
                if price_above_hma_prev and price_near_hma:
                    pullback_long = True
            
            # Short pullback: was below HMA, now near/touching HMA, still below SMA200
            elif hma_bear and htf_1d_bear:
                price_below_hma_prev = close[i-1] < hma_6h[i-1] if not np.isnan(hma_6h[i-1]) else False
                price_near_hma = abs(close[i] - hma_6h[i]) < 0.015 * close[i]
                if price_below_hma_prev and price_near_hma:
                    pullback_short = True
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_6h_fast[i]) and not np.isnan(hma_6h_fast[i-1]):
            if not np.isnan(hma_6h[i]) and not np.isnan(hma_6h[i-1]):
                if hma_6h_fast[i-1] <= hma_6h[i-1] and hma_6h_fast[i] > hma_6h[i]:
                    hma_cross_long = True
                if hma_6h_fast[i-1] >= hma_6h[i-1] and hma_6h_fast[i] < hma_6h[i]:
                    hma_cross_short = True
        
        # === RSI EXTREMES (WIDER BANDS FOR MORE TRADES) ===
        rsi_oversold = rsi_fast[i] < 15.0  # Wider than typical 25/30
        rsi_overbought = rsi_fast[i] > 85.0  # Wider than typical 70/75
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (SIMPLIFIED - MAX 3 CONDITIONS) ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (pullback entries with HTF alignment)
        if is_trending:
            # Long: 1d bull + pullback OR cross + above SMA200
            if htf_1d_bull and above_sma200:
                if pullback_long or hma_cross_long:
                    desired_signal = SIZE_STRONG
            
            # Short: 1d bear + pullback OR cross + below SMA200
            elif htf_1d_bear and below_sma200:
                if pullback_short or hma_cross_short:
                    desired_signal = -SIZE_STRONG
        
        # REGIME 2: CHOPPY (RSI mean reversion - SIMPLE)
        elif is_choppy:
            # Long: RSI oversold + above SMA200 (2 conditions only!)
            if rsi_oversold and above_sma200:
                desired_signal = SIZE_BASE
            
            # Short: RSI overbought + below SMA200 (2 conditions only!)
            elif rsi_overbought and below_sma200:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * atr[i]
                else:
                    stop_price = entry_price + 2.5 * atr[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals