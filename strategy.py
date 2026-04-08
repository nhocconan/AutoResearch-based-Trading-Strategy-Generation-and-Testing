#!/usr/bin/env python3
# 12h_camarilla_pivot_1d_volume_v1
# Hypothesis: Camarilla pivot levels on 1-day timeframe act as strong support/resistance.
# Long when price touches or breaks above S3 with volume > 1.3x average, short when touches or breaks below R3 with volume > 1.3x average.
# Exit when price returns to the daily Pivot Point (mean reversion to mean).
# Uses 12h timeframe for lower frequency trading to reduce fee drag, with 1d pivot levels for structure.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Typical price for pivot calculation
    tp_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    
    # Pivot point
    pivot_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    
    # Ranges
    range_1d = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    r3_1d = pivot_1d + range_1d * 1.1 / 2
    s3_1d = pivot_1d - range_1d * 1.1 / 2
    pivot_1d_arr = pivot_1d.values
    
    # Align to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d.values)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d.values)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_arr)
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below pivot point (mean reversion)
            if close[i] <= pivot_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above pivot point (mean reversion)
            if close[i] >= pivot_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.3x average volume
            volume_ok = volume[i] > 1.3 * avg_volume[i]
            
            # Entry conditions
            if close[i] >= s3_1d_aligned[i] and volume_ok:
                # Additional confirmation: price was below S3 in previous bar
                if i > 0 and close[i-1] < s3_1d_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
            elif close[i] <= r3_1d_aligned[i] and volume_ok:
                # Additional confirmation: price was above R3 in previous bar
                if i > 0 and close[i-1] > r3_1d_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals