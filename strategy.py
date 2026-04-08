#!/usr/bin/env python3
# 12h_camilla_pivot_breakout_volume_v2
# Hypothesis: Uses Camarilla pivot levels from daily timeframe combined with volume surge to trade breakouts.
# Long when price breaks above resistance level R3 with volume > 2x average.
# Short when price breaks below support level S3 with volume > 2x average.
# Exit when price returns to the pivot point (CP) or volume drops below average.
# Uses daily pivot levels to avoid noise, suitable for both trending and ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

name = "12h_camilla_pivot_breakout_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 2x 24-period average (2 days of 12h data)
    vol_ma_period = 24
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 2.0 * vol_ma[i]
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Pivot point (CP) = (High + Low + Close) / 3
    # Resistance levels: R1 = CP + (High - Low) * 1.1/12, R2 = CP + (High - Low) * 1.1/6, R3 = CP + (High - Low) * 1.1/4
    # Support levels: S1 = CP - (High - Low) * 1.1/12, S2 = CP - (High - Low) * 1.1/6, S3 = CP - (High - Low) * 1.1/4
    cp = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    r3 = cp + rng * 1.1 / 4.0
    s3 = cp - rng * 1.1 / 4.0
    
    # Align the daily pivot levels to 12h timeframe
    cp_aligned = align_ltf_to_hlf(prices, df_1d, cp)
    r3_aligned = align_ltf_to_hlf(prices, df_1d, r3)
    s3_aligned = align_ltf_to_hlf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = vol_ma_period  # Wait for volume MA to be valid
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to pivot point or volume drops below average
            if close[i] <= cp_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to pivot point or volume drops below average
            if close[i] >= cp_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above R3 with volume surge
            if close[i] > r3_aligned[i] and vol_surge[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below S3 with volume surge
            elif close[i] < s3_aligned[i] and vol_surge[i]:
                position = -1
                signals[i] = -0.25
    
    return signals