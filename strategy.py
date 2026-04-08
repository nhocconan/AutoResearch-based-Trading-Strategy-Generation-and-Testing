#!/usr/bin/env python3
# 6h_camarilla_pivot_reversal
# Hypothesis: Uses 6-hour price action with 1-day Camarilla pivot levels for mean reversion in range-bound markets.
# Enters short near R3/R4 and long near S3/S4 with volume confirmation, expecting price to revert toward the Pivot (PP).
# Works in both bull and bear markets by fading extremes at statistically significant levels.
# Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_reversal"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1/12)
    # S2 = C - (Range * 1.1/6)
    # S3 = C - (Range * 1.1/4)
    # S4 = C - (Range * 1.1/2)
    # R1 = C + (Range * 1.1/12)
    # R2 = C + (Range * 1.1/6)
    # R3 = C + (Range * 1.1/4)
    # R4 = C + (Range * 1.1/2)
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    s3_1d = close_1d - (range_1d * 1.1 / 4.0)
    s4_1d = close_1d - (range_1d * 1.1 / 2.0)
    r3_1d = close_1d + (range_1d * 1.1 / 4.0)
    r4_1d = close_1d + (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    
    # Volume confirmation (24-period average on 6h = 6 days)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 24
    
    for i in range(start_idx, n):
        if np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price reaches Pivot Point (mean reversion target) or stops making progress
            if close[i] >= pp_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Pivot Point (mean reversion target)
            if close[i] <= pp_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry at extreme levels
            if volume_ok:
                # Long entry: price at or below S3/S4 with rejection
                if close[i] <= s3_1d_aligned[i] and close[i] > low[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price at or above R3/R4 with rejection
                elif close[i] >= r3_1d_aligned[i] and close[i] < high[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals