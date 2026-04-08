#!/usr/bin/env python3
# 6h_weekly_pivot_volume_breakout_v1
# Hypothesis: Trade weekly pivot breakouts on 6h timeframe with volume confirmation.
# Long when: 6h close breaks above weekly R4 pivot AND volume > 1.5x 20-period average.
# Short when: 6h close breaks below weekly S4 pivot AND volume > 1.5x 20-period average.
# Exit when price crosses back below/above weekly pivot point (R1/S1) or volume drops below average.
# Uses weekly pivot levels from 1w timeframe for structural levels, volume for confirmation.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_breakout_v1"
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
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)
    
    pivot_1w = np.full(len(close_1w), np.nan)
    r1_1w = np.full(len(close_1w), np.nan)
    s1_1w = np.full(len(close_1w), np.nan)
    r2_1w = np.full(len(close_1w), np.nan)
    s2_1w = np.full(len(close_1w), np.nan)
    r3_1w = np.full(len(close_1w), np.nan)
    s3_1w = np.full(len(close_1w), np.nan)
    r4_1w = np.full(len(close_1w), np.nan)
    s4_1w = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        if not np.isnan(high_1w[i]) and not np.isnan(low_1w[i]) and not np.isnan(close_1w[i]):
            pivot = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
            pivot_1w[i] = pivot
            r1_1w[i] = 2 * pivot - low_1w[i]
            s1_1w[i] = 2 * pivot - high_1w[i]
            r2_1w[i] = pivot + (high_1w[i] - low_1w[i])
            s2_1w[i] = pivot - (high_1w[i] - low_1w[i])
            r3_1w[i] = high_1w[i] + 2 * (pivot - low_1w[i])
            s3_1w[i] = low_1w[i] - 2 * (high_1w[i] - pivot)
            r4_1w[i] = r3_1w[i] + (high_1w[i] - low_1w[i])
            s4_1w[i] = s3_1w[i] - (high_1w[i] - low_1w[i])
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, 1)  # Need volume MA and at least one weekly bar
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma[i]) or np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below weekly S1 or volume drops below average
            if close[i] < s1_1w_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above weekly R1 or volume drops below average
            if close[i] > r1_1w_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above weekly R4 with volume surge
            if (close[i] > r4_1w_aligned[i] and vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below weekly S4 with volume surge
            elif (close[i] < s4_1w_aligned[i] and vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals