#!/usr/bin/env python3
# 6h_12h1d_camarilla_pivot_v1
# Hypothesis: Use daily Camarilla pivot levels for mean reversion at R3/S3 and breakout continuation at R4/S4.
# In ranging markets, price reverts to mean from extreme levels (R3/S3). In trending markets,
# breaks of R4/S4 with volume confirmation indicate continuation. Works in both bull and bear regimes
# by adapting to market structure via pivot levels and volume filter.
# Target: 15-35 trades/year on 6h timeframe with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h1d_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_ = high_1d - low_1d
    
    # Camarilla levels
    r4 = close_1d + range_ * 1.1 / 2
    r3 = close_1d + range_ * 1.1 / 4
    s3 = close_1d - range_ * 1.1 / 4
    s4 = close_1d - range_ * 1.1 / 2
    
    # Align levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 6h volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 (mean reversion) OR stop at S4 breakdown
            if close[i] < s3_6h[i] or (close[i] < s4_6h[i] and not vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (mean reversion) OR stop at R4 breakdown
            if close[i] > r3_6h[i] or (close[i] > r4_6h[i] and not vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bounce from S3 with volume surge (mean reversion)
            if close[i] > s3_6h[i] and close[i] <= s3_6h[i] * 1.005 and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: rejection at R3 with volume surge (mean reversion)
            elif close[i] < r3_6h[i] and close[i] >= r3_6h[i] * 0.995 and vol_surge:
                position = -1
                signals[i] = -0.25
            # Long breakout: break above R4 with volume surge (continuation)
            elif close[i] > r4_6h[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short breakout: break below S4 with volume surge (continuation)
            elif close[i] < s4_6h[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals