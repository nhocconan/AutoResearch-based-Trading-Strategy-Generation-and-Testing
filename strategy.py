#!/usr/bin/env python3
"""
6h_1w_elliott_wave_fib_v1
Strategy: 6h Elliott Wave-inspired Fibonacci retracement with weekly trend filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: Combines Fibonacci retracement levels from weekly swing points with 60-period EMA trend filter on 6h. Enters long at 0.618 Fib support in uptrend, short at 0.382 Fib resistance in downtrend. Uses weekly swing high/low to define the trend leg, avoiding chop. Designed to work in both bull (buy dips) and bear (sell rallies) by following the higher timeframe trend. Low trade frequency expected due to strict Fib + trend confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_elliott_wave_fib_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 60-period EMA on 6h for trend filter
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Weekly swing points: find highest high and lowest low over lookback
    lookback = 12  # ~3 months of weekly data
    roll_max = pd.Series(df_1w['high']).rolling(window=lookback, min_periods=lookback).max().values
    roll_min = pd.Series(df_1w['low']).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align weekly swing points to 6h
    swing_high_aligned = align_htf_to_ltf(prices, df_1w, roll_max)
    swing_low_aligned = align_htf_to_ltf(prices, df_1w, roll_min)
    
    # Calculate Fibonacci levels: 0.382 and 0.618
    swing_range = swing_high_aligned - swing_low_aligned
    fib_0382 = swing_low_aligned + 0.382 * swing_range
    fib_0618 = swing_low_aligned + 0.618 * swing_range
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_60[i]) or np.isnan(swing_high_aligned[i]) or 
            np.isnan(swing_low_aligned[i]) or np.isnan(fib_0382[i]) or 
            np.isnan(fib_0618[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 60 EMA
        uptrend = price_close > ema_60[i]
        downtrend = price_close < ema_60[i]
        
        # Long at 0.618 Fib support in uptrend
        long_signal = uptrend and price_close <= fib_0618[i]
        
        # Short at 0.382 Fib resistance in downtrend
        short_signal = downtrend and price_close >= fib_0382[i]
        
        # Exit when price crosses the 60 EMA
        exit_long = position == 1 and price_close < ema_60[i]
        exit_short = position == -1 and price_close > ema_60[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals