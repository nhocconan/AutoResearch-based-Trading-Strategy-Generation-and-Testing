#!/usr/bin/env python3
# 6H_1D_1W_Supertrend_Trend_Follow
# Hypothesis: On 6h timeframe, follow the trend using Supertrend(10,3) on 1d for direction and Supertrend(10,3) on 1w for regime filter.
# Enter long when price > 6h Supertrend, 1d Supertrend uptrend, and 1w Supertrend uptrend.
# Enter short when price < 6h Supertrend, 1d Supertrend downtrend, and 1w Supertrend downtrend.
# Exit when price crosses back through 6h Supertrend or higher timeframe trend changes.
# This multi-timeframe trend alignment should reduce whipsaws and work in both bull and bear markets by only trading with the dominant trend.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6H_1D_1W_Supertrend_Trend_Follow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def supertrend(high, low, close, period=10, multiplier=3):
    """Calculate Supertrend indicator."""
    # Calculate ATR
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.DataFrame(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close, np.nan, dtype=float)
    direction = np.full_like(close, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    # Start from period to have enough data for ATR
    for i in range(period, len(close)):
        if i == period:
            # First valid value
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            # Determine trend direction
            if close[i-1] > supertrend[i-1]:
                # Was in uptrend
                if close[i] <= upper_band[i]:
                    supertrend[i] = supertrend[i-1]
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1
            else:
                # Was in downtrend
                if close[i] >= lower_band[i]:
                    supertrend[i] = supertrend[i-1]
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = 1
                    
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Supertrend for 6h (entry trigger)
    st_6h, dir_6h = supertrend(high, low, close, period=10, multiplier=3)
    
    # Calculate Supertrend for 1d (trend filter)
    st_1d, dir_1d = supertrend(high_1d, low_1d, close_1d, period=10, multiplier=3)
    
    # Calculate Supertrend for 1w (regime filter)
    st_1w, dir_1w = supertrend(high_1w, low_1w, close_1w, period=10, multiplier=3)
    
    # Align 1d and 1w indicators to 6h
    st_1d_aligned = align_htf_to_ltf(prices, df_1d, st_1d)
    dir_1d_aligned = align_htf_to_ltf(prices, df_1d, dir_1d.astype(float))
    st_1w_aligned = align_htf_to_ltf(prices, df_1w, st_1w)
    dir_1w_aligned = align_htf_to_ltf(prices, df_1w, dir_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(30, 10)  # Ensure we have enough data for 6h Supertrend
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(st_6h[i]) or np.isnan(st_1d_aligned[i]) or np.isnan(dir_1d_aligned[i]) or
            np.isnan(st_1w_aligned[i]) or np.isnan(dir_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above 6h Supertrend, 1d uptrend, 1w uptrend
            if close[i] > st_6h[i] and dir_1d_aligned[i] > 0 and dir_1w_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: price below 6h Supertrend, 1d downtrend, 1w downtrend
            elif close[i] < st_6h[i] and dir_1d_aligned[i] < 0 and dir_1w_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 6h Supertrend or higher timeframe trend turns down
            if close[i] < st_6h[i] or dir_1d_aligned[i] < 0 or dir_1w_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 6h Supertrend or higher timeframe trend turns up
            if close[i] > st_6h[i] or dir_1d_aligned[i] > 0 or dir_1w_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals