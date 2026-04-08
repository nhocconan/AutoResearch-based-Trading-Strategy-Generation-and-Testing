#!/usr/bin/env python3
"""
6h Weekly Pivot + Volume Confirmation Strategy
Hypothesis: Weekly pivot points act as institutional support/resistance. 
When price breaks above weekly R3 with volume confirmation, it signals institutional buying (long).
When price breaks below weekly S3 with volume confirmation, it signals institutional selling (short).
Weekly timeframe filters noise, 6h provides timely entries. Works in both bull/bear markets as
pivots adapt to volatility. Target: 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = 2*P - L, R2 = P + (H - L), R3 = H + 2*(P - L)
    # S1 = 2*P - H, S2 = P - (H - L), S3 = L - 2*(H - P)
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly data to 6h timeframe (with shift(1) for look-ahead bias prevention)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly pivot OR volume dries up
            if close[i] < pivot_aligned[i] or not vol_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above weekly pivot OR volume dries up
            if close[i] > pivot_aligned[i] or not vol_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above weekly R3 with volume spike
            if (close[i] > r3_aligned[i] and vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly S3 with volume spike
            elif (close[i] < s3_aligned[i] and vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals