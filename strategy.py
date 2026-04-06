# 6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
# Hypothesis: 6h timeframe with weekly pivot direction filter reduces whipsaw by aligning with major trend.
# Weekly pivot provides strong support/resistance levels from institutional activity.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) suitable for 6h timeframe.
# Works in both bull and bear markets by using pivot direction as trend filter.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Weekly pivot points (using previous week's data)
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot points: P = (H + L + C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Calculate support/resistance levels
    r1 = 2 * weekly_pivot - weekly_low
    s1 = 2 * weekly_pivot - weekly_high
    r2 = weekly_pivot + (weekly_high - weekly_low)
    s2 = weekly_pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Donchian midline
        if position == 1:  # long position
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with pivot direction and volume confirmation
            # Long: price breaks above Donchian upper AND price > weekly R3 AND volume confirmation
            if (close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and 
                close[i] > r3_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < weekly S3 AND volume confirmation
            elif (close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and 
                  close[i] < s3_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals