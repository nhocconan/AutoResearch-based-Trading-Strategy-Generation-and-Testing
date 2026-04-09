#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_spike_v6
# Hypothesis: 4h strategy using 1d Camarilla pivot levels with volume spike confirmation.
# Long: Price breaks above H4 pivot with volume > 2.0x 20-period average
# Short: Price breaks below L4 pivot with volume > 2.0x 20-period average
# Exit: Price returns to H3/L3 levels
# Uses 4h primary timeframe with 1d HTF for Camarilla pivot calculation.
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and fee drag.
# Volume spike filter reduces false breakouts. Works in both bull and bear markets by
# capturing institutional breakouts with confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_spike_v6"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (High + Low + Close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    # Camarilla levels
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    h4_1d = pivot_1d + (range_1d * 1.1 / 2)
    l4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align 1d Camarilla levels to 4h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period for all indicators
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (strict filter)
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to H3 level
            if close[i] <= h3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to L3 level
            if close[i] >= l3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above H4 with volume confirmation
            if close[i] > h4_1d_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L4 with volume confirmation
            elif close[i] < l4_1d_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals