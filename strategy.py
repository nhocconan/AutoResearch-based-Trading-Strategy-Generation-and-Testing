#!/usr/bin/env python3
"""
6h_LongTermTrend_WeeklyPivot_Breakout
Hypothesis: Use weekly pivot points (from 1w data) to define major support/resistance zones.
            Enter long when price breaks above weekly R3 with 6h EMA(20) uptrend and volume spike.
            Enter short when price breaks below weekly S3 with 6h EMA(20) downtrend and volume spike.
            Exit when price crosses weekly pivot point (mean of R3/S3) or EMA trend flips.
            Weekly pivot provides structural levels that work in both trending and ranging markets.
            EMA(20) filters for trend alignment, volume spike confirms breakout strength.
            Designed for 6h timeframe to capture multi-day moves with low frequency (~15-30 trades/year).
"""
name = "6h_LongTermTrend_WeeklyPivot_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's HLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly R3, S3, and pivot (central) levels
    # Classic formula: R3 = Close + 1.1*(High - Low), S3 = Close - 1.1*(High - Low)
    weekly_range = high_1w - low_1w
    weekly_r3 = close_1w + (weekly_range * 1.1)
    weekly_s3 = close_1w - (weekly_range * 1.1)
    weekly_pivot = (weekly_r3 + weekly_s3) * 0.5  # midpoint
    
    # Align weekly levels to 6h timeframe (use prior week's levels)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # 6h EMA(20) for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_prev = np.roll(ema_20, 1)
    ema_20_prev[0] = ema_20[0]
    ema_rising = ema_20 > ema_20_prev
    ema_falling = ema_20 < ema_20_prev
    
    # Volume spike: current volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(ema_rising[i]) or
            np.isnan(ema_falling[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > weekly R3 + EMA uptrend + volume spike
            if (close[i] > weekly_r3_aligned[i] and 
                ema_rising[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < weekly S3 + EMA downtrend + volume spike
            elif (close[i] < weekly_s3_aligned[i] and 
                  ema_falling[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly pivot OR EMA turns down
            if (close[i] < weekly_pivot_aligned[i]) or (not ema_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly pivot OR EMA turns up
            if (close[i] > weekly_pivot_aligned[i]) or (not ema_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals