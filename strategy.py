#!/usr/bin/env python3
# Hypothesis: 4h Camarilla Pivot R3/S3 breakout with 1-day EMA34 trend filter and volume confirmation.
# Camarilla pivot levels (R3/S3) provide strong intraday support/resistance levels.
# Breakout above R3 or below S3 with volume confirmation indicates strong momentum.
# 1-day EMA34 filter ensures we only trade in the direction of the daily trend.
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear markets by following the daily trend direction.

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
    
    # Get daily data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34 calculation
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3/S3 are the most significant
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    
    # Calculate daily EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA34 to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above EMA34 = uptrend, below = downtrend
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > r3_aligned[i]  # Break above R3
        breakout_short = close[i] < s3_aligned[i]  # Break below S3
        
        # Entry conditions with volume confirmation
        long_entry = uptrend and breakout_long and volume_filter[i]
        short_entry = downtrend and breakout_short and volume_filter[i]
        
        # Exit conditions: when price returns to pivot level or trend reverses
        pivot_level = (df_1d['high'].iloc[-1] + df_1d['low'].iloc[-1] + df_1d['close'].iloc[-1]) / 3 if len(df_1d) > 0 else 0
        # Simplified exit: reverse signal or trend change
        long_exit = not uptrend or close[i] < pivot_level
        short_exit = not downtrend or close[i] > pivot_level
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0