#!/usr/bin/env python3
"""
Experiment #111: 6h Primary + 1w/1d HTF — ADX Trend Filter + HMA + RSI Pullback

Hypothesis: 6h strategies failed because they traded through choppy/whipsaw periods.
Key insight from 102 failed experiments: regime detection is critical, but complex
regime switches (Chop, CRIS, Fisher) added noise. Simpler ADX trend filter works better.

Design:
- 1w HMA: Major trend bias (only trade with weekly trend)
- 1d HMA: Intermediate trend confirmation
- 6h ADX(14): Filter out choppy markets (ADX<20 = flat, no trades)
- 6h HMA(21): Entry trend confirmation
- 6h RSI(14): Pullback entry (long when RSI 35-50 in uptrend, short 50-65 in downtrend)
- ATR(14) 2.5x trailing stoploss

Why this should work:
- ADX filter eliminates 6h whipsaw (main killer of 6h strategies)
- Weekly HMA prevents counter-trend trades in major moves
- RSI pullback entries avoid chasing breakouts (failed pattern in exp#100,102,108)
- Conservative size (0.28) limits drawdown during 2022 crash
- Should generate 30-60 trades/year (ADX>20 happens ~40% of time)

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_adx_hma_rsi_1w1d_v1"
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
    """Average True Range for stoploss"""
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
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = trending, ADX < 20 = choppy/range
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        else:
            plus_dm[i] = 0.0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
        else:
            minus_dm[i] = 0.0
    
    # Smooth with EMA
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1w + 1d HMA) ===
        # Bullish: price above both weekly and daily HMA
        # Bearish: price below both weekly and daily HMA
        htf_bull = close[i] > hma_1w_aligned[i] and close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i] and close[i] < hma_1d_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === TREND STRENGTH FILTER (ADX) ===
        # ADX > 22 = trending (take trades)
        # ADX < 18 = choppy (no trades, or exit existing)
        is_trending = adx[i] > 22.0
        is_choppy = adx[i] < 18.0
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI pulled back to 35-50 in uptrend (not oversold, just cooling)
        # Short: RSI pulled back to 50-65 in downtrend (not overbought, just cooling)
        rsi_pullback_long = 35.0 <= rsi[i] <= 52.0
        rsi_pullback_short = 48.0 <= rsi[i] <= 65.0
        
        # Strong RSI confirmation
        rsi_bull_strong = rsi[i] > 45.0 and rsi[i] < 70.0
        rsi_bear_strong = rsi[i] < 55.0 and rsi[i] > 30.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Only trade when ADX indicates trending market
        if is_trending:
            # LONG: HTF bull + 6h HMA bull + RSI pullback
            if htf_bull and hma_bull and rsi_pullback_long:
                desired_signal = SIZE
            elif htf_bull and hma_bull and rsi_bull_strong and adx[i] > 28.0:
                # Strong trend, less strict RSI
                desired_signal = SIZE * 0.8
            
            # SHORT: HTF bear + 6h HMA bear + RSI pullback
            elif htf_bear and hma_bear and rsi_pullback_short:
                desired_signal = -SIZE
            elif htf_bear and hma_bear and rsi_bear_strong and adx[i] > 28.0:
                # Strong trend, less strict RSI
                desired_signal = -SIZE * 0.8
        
        # If choppy and in position, consider exiting
        if is_choppy and in_position:
            # Reduce position in choppy market
            if position_side > 0 and rsi[i] > 55.0:
                desired_signal = 0.0
            elif position_side < 0 and rsi[i] < 45.0:
                desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals