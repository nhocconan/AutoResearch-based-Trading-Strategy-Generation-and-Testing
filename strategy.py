#!/usr/bin/env python3
name = "4h_WeeklyPivot_Strategy"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    # Pivot Point (PP) = (High + Low + Close) / 3
    # Resistance 1 (R1) = (2 * PP) - Low
    # Support 1 (S1) = (2 * PP) - High
    pp_weekly = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    r1_weekly = (2 * pp_weekly) - df_1w['low']
    s1_weekly = (2 * pp_weekly) - df_1w['high']
    
    # Align weekly pivot levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_weekly.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_weekly.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_weekly.values)
    
    # Volume filter: volume above 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above S1 with volume confirmation
            if close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 with volume confirmation
            elif close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below PP or opposite signal
            if close[i] < pp_aligned[i] or (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above PP or opposite signal
            if close[i] > pp_aligned[i] or (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals