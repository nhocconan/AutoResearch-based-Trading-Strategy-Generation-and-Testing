#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_volume_v1
# Hypothesis: 6h strategy using weekly Camarilla pivot levels for direction and 6h Donchian(20) breakouts for entry.
# In bull markets, price breaks above weekly R4 with volume; in bear markets, breaks below S4 with volume.
# Weekly pivot provides higher timeframe bias, Donchian breakout captures momentum, volume filters false signals.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for weekly Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point
    pw = (high_1w + low_1w + close_1w) / 3
    r1 = pw + (high_1w - low_1w) * 1.1 / 12
    s1 = pw - (high_1w - low_1w) * 1.1 / 12
    r2 = pw + (high_1w - low_1w) * 1.1 / 6
    s2 = pw - (high_1w - low_1w) * 1.1 / 6
    r3 = pw + (high_1w - low_1w) * 1.1 / 4
    s3 = pw - (high_1w - low_1w) * 1.1 / 4
    r4 = pw + (high_1w - low_1w) * 1.1 / 2
    s4 = pw - (high_1w - low_1w) * 1.1 / 2
    
    # Align weekly pivot levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves below 6h Donchian low or weekly S3
            if close[i] < low_20[i] or close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above 6h Donchian high or weekly R3
            if close[i] > high_20[i] or close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above weekly R4 with volume
                if close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below weekly S4 with volume
                elif close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals