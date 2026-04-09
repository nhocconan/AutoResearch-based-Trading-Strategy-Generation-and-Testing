#!/usr/bin/env python3
# 6h_donchian_12h_pivot_volume_v1
# Hypothesis: 6h Donchian(20) breakouts with 12h Camarilla pivot direction filter and volume confirmation work in both bull and bear markets.
# Uses 6h Donchian channel breakouts for entry, 12h Camarilla pivot levels for direction bias (long above PP, short below PP),
# and volume > 1.5x 20-period average for confirmation. Designed for low frequency (12-37 trades/year) to avoid fee drag.
# Works in bull markets via breakout continuation and in bear markets via mean reversion at extreme pivot levels (R4/S4).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_12h_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    # 12h Camarilla pivot levels (calculated from prior 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Prior 12h bar's high, low, close for pivot calculation
    phigh = df_12h['high'].shift(1).values  # previous 12h bar high
    plow = df_12h['low'].shift(1).values    # previous 12h bar low
    pclose = df_12h['close'].shift(1).values # previous 12h bar close
    
    # Camarilla pivot calculations
    pivot = (phigh + plow + pclose) / 3.0
    range_val = phigh - plow
    r4 = pclose + range_val * 1.1 / 2
    r3 = pclose + range_val * 1.1 / 4
    s3 = pclose - range_val * 1.1 / 4
    s4 = pclose - range_val * 1.1 / 2
    
    # Align 12h pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 6h Donchian lower band or price < S3 (mean reversion)
            if close[i] < low_min[i] or close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 6h Donchian upper band or price > R3 (mean reversion)
            if close[i] > high_max[i] or close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: 6h Donchian breakout above upper band with volume confirmation and price > pivot
            if close[i] > high_max[i] and volume[i] > vol_threshold[i] and close[i] > pivot_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: 6h Donchian breakout below lower band with volume confirmation and price < pivot
            elif close[i] < low_min[i] and volume[i] > vol_threshold[i] and close[i] < pivot_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals