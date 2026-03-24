#!/usr/bin/env python3
"""
Experiment #095: 1h Primary + 4h/1d HTF — Triple HMA Trend Confluence

Hypothesis: After analyzing 80+ failed experiments, the pattern for 1h is clear:
- Session filters cause 0 trades (#088 failed with session 8-20 UTC)
- Choppiness/Connors RSI too restrictive (#087, #093 failed)
- Simple HMA + loose RSI works on 12h/1d (#083, #086 kept)

For 1h to work, I need HTF trend alignment WITHOUT over-filtering:
1. 1d HMA50 = major trend bias (price above/below)
2. 4h HMA16/48 = intermediate trend (crossover direction)
3. 1h HMA16/48 = entry trigger (crossover + RSI confirmation)
4. RSI(14) loose filter: >40 for long, <60 for short (not 30/70)
5. Bollinger Band position: price in middle 60% of bands (not extremes)
6. ATR trailing stoploss: 2.5x for risk management

Key design choices:
- Timeframe: 1h (target 30-60 trades/year)
- HTF: 1d + 4h for trend alignment (dual HTF confluence)
- RSI thresholds: 40/60 (looser than 30/70 to ensure trades)
- Position size: 0.25 (smaller for 1h noise)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_triple_hma_rsi_bb_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_series = pd.Series(close)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = close_series.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma_full = close_series.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    
    return hma.values

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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands - returns lower, middle, upper"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    close_series = pd.Series(close)
    middle = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    return lower, middle, upper

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_fast_raw = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_slow_raw = calculate_hma(df_4h['close'].values, period=48)
    hma_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_fast_raw)
    hma_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slow_raw)
    
    # Calculate primary (1h) indicators
    hma_1h_fast = calculate_hma(close, period=16)
    hma_1h_slow = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_lower, bb_middle, bb_upper = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 1h)
    
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
        if np.isnan(hma_1h_fast[i]) or np.isnan(hma_1h_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_fast_aligned[i]) or np.isnan(hma_4h_slow_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA crossover) ===
        htf_4h_bull = hma_4h_fast_aligned[i] > hma_4h_slow_aligned[i]
        htf_4h_bear = hma_4h_fast_aligned[i] < hma_4h_slow_aligned[i]
        
        # === 1h TREND (HMA crossover) ===
        hma_1h_cross_bull = hma_1h_fast[i] > hma_1h_slow[i]
        hma_1h_cross_bear = hma_1h_fast[i] < hma_1h_slow[i]
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        rsi_ok_long = rsi[i] > 40.0
        rsi_ok_short = rsi[i] < 60.0
        
        # === BOLLINGER BAND POSITION (middle 60% of bands) ===
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range > 1e-10:
            bb_position = (close[i] - bb_lower[i]) / bb_range
            bb_ok_long = bb_position > 0.2  # not at lower extreme
            bb_ok_short = bb_position < 0.8  # not at upper extreme
        else:
            bb_ok_long = True
            bb_ok_short = True
        
        # === DESIRED SIGNAL (Triple HTF confluence) ===
        # LONG: 1d bull + 4h bull + 1h HMA cross bull + RSI > 40 + BB ok
        # SHORT: 1d bear + 4h bear + 1h HMA cross bear + RSI < 60 + BB ok
        desired_signal = 0.0
        
        if htf_bull and htf_4h_bull and hma_1h_cross_bull and rsi_ok_long and bb_ok_long:
            desired_signal = SIZE
        elif htf_bear and htf_4h_bear and hma_1h_cross_bear and rsi_ok_short and bb_ok_short:
            desired_signal = -SIZE
        
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