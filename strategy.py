#!/usr/bin/env python3
# 6h_1d_weekly_pivot_breakout_v1
# Hypothesis: 6-hour breakout from weekly pivot levels (R4/S4) with 1-day volume confirmation.
# Long when price breaks above weekly R4 with volume > 1.5x 20-period average.
# Short when price breaks below weekly S4 with volume > 1.5x 20-period average.
# Exit when price returns to weekly pivot (PP) or opposite extreme (S4/R4) is breached.
# Uses weekly pivot for structure and 1d volume filter to avoid false breakouts.
# Target: 15-25 trades/year to minimize fee drag while capturing strong moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_breakout_v1"
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
    
    # Calculate 20-period average volume
    vol_avg = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i < 20:
            vol_avg[i] = np.nan
        else:
            if i == 20:
                vol_avg[i] = vol_sum / 20
            else:
                vol_sum -= volume[i-20]
                vol_avg[i] = vol_sum / 20
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP, R1-R4, S1-S4
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    r4 = weekly_high + 3 * (pp - weekly_low)
    s4 = weekly_low - 3 * (weekly_high - pp)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 0 else 0
        vol_confirm = vol_ratio > 1.5
        
        if position == 1:  # Long
            # Exit: price returns to pivot point OR breaks below S4
            if close[i] <= pp_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to pivot point OR breaks above R4
            if close[i] >= pp_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry: long breakout above R4 with volume confirmation
            if close[i] > r4_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Entry: short breakout below S4 with volume confirmation
            elif close[i] < s4_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals