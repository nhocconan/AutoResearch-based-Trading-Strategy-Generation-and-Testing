#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d ADX for trend strength and 6h Supertrend for direction
# Long when 1d ADX > 25 (strong trend) AND 6h Supertrend gives bullish signal
# Short when 1d ADX > 25 (strong trend) AND 6h Supertrend gives bearish signal
# Exit when Supertrend reverses or ADX falls below 20 (weak trend)
# Uses discrete sizing 0.25 to manage risk in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# ADX ensures we only trade in strong trending markets, avoiding chop
# Supertrend provides clear entry/exit signals with ATR-based stops
# Works in bull markets (catch uptrends) and bear markets (catch downtrends)

name = "6h_1dADX_Supertrend_TrendFollow"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX (14+ periods)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (Average Directional Index)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = wilder_smooth(dx, period)
    
    # Align 1d ADX to 6h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Supertrend (ATR=10, mult=3.0)
    # True Range for 6h
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr1_6h[0] = 0
    tr2_6h[0] = 0
    tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    
    # ATR using Wilder's smoothing
    atr_period = 10
    atr_6h = wilder_smooth(tr_6h, atr_period)
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + (3.0 * atr_6h)
    lower_band = hl2 - (3.0 * atr_6h)
    
    # Initialize Supertrend arrays
    supertrend = np.full_like(close, np.nan)
    direction = np.full_like(close, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # First valid Supertrend value
    start_idx = atr_period
    if start_idx < len(close):
        supertrend[start_idx] = upper_band[start_idx]
        direction[start_idx] = -1  # Start in downtrend (price below upper band)
    
    # Calculate Supertrend iteratively
    for i in range(start_idx + 1, len(close)):
        if close[i] > supertrend[i-1]:
            # Potential uptrend
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            # Potential downtrend
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            direction[i] = -1
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(direction[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: strong trend (ADX > 25) AND Supertrend bullish (direction = 1)
            if adx_aligned[i] > 25 and direction[i] == 1:
                signals[i] = 0.25
                position = 1
            # Enter short: strong trend (ADX > 25) AND Supertrend bearish (direction = -1)
            elif adx_aligned[i] > 25 and direction[i] == -1:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Supertrend turns bearish OR trend weakens (ADX < 20)
            if direction[i] == -1 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Supertrend turns bullish OR trend weakens (ADX < 20)
            if direction[i] == 1 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals