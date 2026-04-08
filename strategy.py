#!/usr/bin/env python3
# 12h_camilla_pivot_breakout_volume_v2
# Hypothesis: Camarilla pivot levels from daily timeframe provide institutional support/resistance.
# Breakout above R4 or below S4 with volume > 2x 24-period average signals institutional breakout.
# Exit when price returns to the pivot point (PP) or volume drops below average.
# Uses 12h timeframe with 1d Camarilla pivots for reduced noise and higher success rate.
# Target: 15-30 trades/year to stay within optimal range for 12h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    
    # Get 1d data for Camarilla pivots (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using yesterday's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots: (H+L+C)/3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: S1 = C - (R*1.1/12), S2 = C - (R*1.1/6), S3 = C - (R*1.1/4), S4 = C - (R*1.1/2)
    # R4 = C + (R*1.1/2), R3 = C + (R*1.1/4), R2 = C + (R*1.1/6), R1 = C + (R*1.1/12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    s2 = close_1d - (range_1d * 1.1 / 6)
    s3 = close_1d - (range_1d * 1.1 / 4)
    s4 = close_1d - (range_1d * 1.1 / 2)
    r1 = close_1d + (range_1d * 1.1 / 12)
    r2 = close_1d + (range_1d * 1.1 / 6)
    r3 = close_1d + (range_1d * 1.1 / 4)
    r4 = close_1d + (range_1d * 1.1 / 2)
    
    # Align all levels to 12h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Volume filter: 2x 24-period average (2 days of 12h data)
    vol_ma_period = 24
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 2.0 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(24, 1) + 1  # Need volume MA and at least one day of pivot data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to pivot point or volume drops below average
            if close[i] <= pivot_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to pivot point or volume drops below average
            if close[i] >= pivot_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above R4 with volume surge
            if (close[i] > r4_aligned[i] and vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below S4 with volume surge
            elif (close[i] < s4_aligned[i] and vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals