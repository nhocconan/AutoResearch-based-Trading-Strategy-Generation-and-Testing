#!/usr/bin/env python3
"""
Experiment #404: 12h Primary + 1d/1w HTF — KAMA/Fisher Dual Regime v1

Hypothesis: Previous 12h strategies failed due to overly complex regime detection
(ADX + Choppiness together rarely aligned). This version SIMPLIFIES to ADX-only
regime and adds Fisher Transform for better entry timing in bear/range markets.

Key changes from #352:
1. KAMA instead of HMA - adaptive to volatility, better noise filtering
2. Fisher Transform for entry timing - proven in bear markets (catches reversals)
3. ADX-only regime (remove Choppiness) - simpler, triggers more often
4. RSI thresholds 20/80 (not 25/75) - more extreme = better mean reversion
5. Remove volume confirmation - too restrictive on 12h timeframe
6. Add 1w HTF for major trend bias (not just 1d)

Regime Detection (SIMPLIFIED):
- ADX > 25 = trending → KAMA breakout entries
- ADX <= 25 = choppy/range → Fisher/RSI mean reversion

Entry Logic:
- Trending Long: KAMA bull + 1d KAMA bull + 1w KAMA bull + Fisher > -1.5
- Trending Short: KAMA bear + 1d KAMA bear + 1w KAMA bear + Fisher < +1.5
- Choppy Long: RSI < 20 + Fisher < -1.5 (double oversold)
- Choppy Short: RSI > 80 + Fisher > +1.5 (double overbought)

Position sizing: 0.25 base, 0.30 when all HTF aligned
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.45, DD>-35%, trades>=25 train, trades>=5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_regime_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(slow_period, n):
        price_change = abs(close[i] - close[i - slow_period])
        vol_sum = 0.0
        for j in range(i - slow_period + 1, i + 1):
            vol_sum += abs(close[j] - close[j - 1])
        if vol_sum > 1e-10:
            er[i] = price_change / vol_sum
    
    # Calculate smoothing constant
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    for i in range(slow_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    # Calculate KAMA
    for i in range(slow_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_fisher(close, period=9):
    """Ehlers Fisher Transform - normalizes price for reversal detection"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(close[i - period + 1:i + 1])
        lowest = np.min(close[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            value = 0.66 * ((close[i] - lowest) / price_range - 0.5) + 0.67 * (
                0.66 * ((close[i - 1] - lowest) / price_range - 0.5) + 0.67 * (
                    0.66 * ((close[i - 2] - lowest) / price_range - 0.5)
                ) if i >= 2 else 0
            )
            value = max(min(value, 0.999), -0.999)
            fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
    
    return fisher

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=21)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    fisher = calculate_fisher(close, period=9)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_12h[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (ADX only - simplified) ===
        is_trending = adx[i] > 25.0
        is_choppy = adx[i] <= 25.0
        
        # === HTF BIAS (1d and 1w) ===
        htf_1d_bull = close[i] > kama_1d_aligned[i]
        htf_1d_bear = close[i] < kama_1d_aligned[i]
        htf_1w_bull = close[i] > kama_1w_aligned[i]
        htf_1w_bear = close[i] < kama_1w_aligned[i]
        
        # === 12h KAMA TREND ===
        kama_bull = close[i] > kama_12h[i]
        kama_bear = close[i] < kama_12h[i]
        
        # === KAMA CROSSOVER ===
        kama_cross_long = False
        kama_cross_short = False
        if i > 0:
            prev_kama = kama_12h[i - 1]
            curr_kama = kama_12h[i]
            prev_close = close[i - 1]
            curr_close = close[i]
            
            if not np.isnan(prev_kama) and not np.isnan(curr_kama):
                if prev_close <= prev_kama and curr_close > curr_kama:
                    kama_cross_long = True
                if prev_close >= prev_kama and curr_close < curr_kama:
                    kama_cross_short = True
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === RSI EXTREMES (LOOSENED for more trades) ===
        rsi_oversold = rsi[i] < 20.0
        rsi_overbought = rsi[i] > 80.0
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer conditions) ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (breakout + full HTF alignment)
        if is_trending:
            # Long: KAMA bull + 1d bull + (1w bull OR neutral) + Fisher confirmation
            if kama_bull and htf_1d_bull:
                if fisher_oversold or kama_cross_long:
                    # Check 1w alignment for stronger signal
                    if htf_1w_bull:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            # Short: KAMA bear + 1d bear + (1w bear OR neutral) + Fisher confirmation
            elif kama_bear and htf_1d_bear:
                if fisher_overbought or kama_cross_short:
                    # Check 1w alignment for stronger signal
                    if htf_1w_bear:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        # REGIME 2: CHOPPY (RSI + Fisher mean reversion - DOUBLE confirmation)
        elif is_choppy:
            # Long: RSI oversold + Fisher oversold (double confirmation)
            if rsi_oversold and fisher_oversold and above_sma200:
                desired_signal = SIZE_BASE
            
            # Short: RSI overbought + Fisher overbought (double confirmation)
            elif rsi_overbought and fisher_overbought and below_sma200:
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
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals