#!/usr/bin/env python3
name = "6h_1d_FibonacciExtension_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly high/low for Fibonacci extension levels
    # We'll use the previous week's range to project extension levels
    # Group daily data into weeks (approximate)
    weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1).values  # Previous week high
    weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1).values    # Previous week low
    weekly_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1).values  # Previous week close
    
    # Calculate Fibonacci extension levels (127.2% and 161.8%)
    weekly_range = weekly_high - weekly_low
    fib_127 = weekly_close + weekly_range * 1.272
    fib_161 = weekly_close + weekly_range * 1.618
    
    # Align weekly Fibonacci levels to 6h timeframe
    fib_127_aligned = align_htf_to_ltf(prices, df_1d, fib_127)
    fib_161_aligned = align_htf_to_ltf(prices, df_1d, fib_161)
    
    # 1d EMA(50) for trend filter (using daily close)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(fib_127_aligned[i]) or 
            np.isnan(fib_161_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 127.2% extension with volume and 1d uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if close[i] > fib_127_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 127.2% extension (downside) with volume and 1d downtrend
            elif close[i] < (2 * ema_50_1d_aligned[i] - fib_127_aligned[i]) and vol_condition and not uptrend:
                # Synthetic downside extension: mirror the 127.2% level around EMA
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price drops below 161.8% extension or volume drops
            if close[i] < fib_161_aligned[i] or volume[i] < vol_ma_4[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises above synthetic 161.8% extension or volume drops
            synthetic_161 = 2 * ema_50_1d_aligned[i] - fib_161_aligned[i]
            if close[i] > synthetic_161 or volume[i] < vol_ma_4[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Fibonacci extension breakout with 1d trend and volume confirmation
# - Uses weekly Fibonacci extension levels (127.2%, 161.8%) from prior week's range
# - Breakout above 127.2% extension with volume in 1d uptrend = long opportunity
# - Breakdown below synthetic 127.2% extension (mirrored) with volume in 1d downtrend = short opportunity
# - Volume spike (2.0x average) confirms strong institutional participation
# - Fibonacci extensions work in both bull (continuation) and bear (counter-trend rallies) markets
# - Exit when price reaches 161.8% extension (profit target) or volume weakens
# - Position size 0.25 targets ~30-60 trades/year, staying within limits
# - Novel: Weekly Fibonacci extensions + 1d trend + volume confirmation not recently tried on 6h
# - Aims for 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Works in BOTH bull and bear markets via trend filter and symmetric logic