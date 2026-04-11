#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # H1 = C + 1.1 * (H - L) / 12
    # L1 = C - 1.1 * (H - L) / 12
    # H2 = C + 1.1 * (H - L) / 6
    # L2 = C - 1.1 * (H - L) / 6
    # H3 = C + 1.1 * (H - L) / 4
    # L3 = C - 1.1 * (H - L) / 4
    # H4 = C + 1.1 * (H - L) / 2
    # L4 = C - 1.1 * (H - L) / 2
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot and levels for each 12h bar
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Resistance levels
    h1_12h = close_12h + 1.1 * range_12h / 12
    h2_12h = close_12h + 1.1 * range_12h / 6
    h3_12h = close_12h + 1.1 * range_12h / 4
    h4_12h = close_12h + 1.1 * range_12h / 2
    
    # Support levels
    l1_12h = close_12h - 1.1 * range_12h / 12
    l2_12h = close_12h - 1.1 * range_12h / 6
    l3_12h = close_12h - 1.1 * range_12h / 4
    l4_12h = close_12h - 1.1 * range_12h / 2
    
    # Align 12h levels to 4h (use previous 12h bar's values to avoid look-ahead)
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    h1_aligned = align_htf_to_ltf(prices, df_12h, h1_12h)
    h2_aligned = align_htf_to_ltf(prices, df_12h, h2_12h)
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    h4_aligned = align_htf_to_ltf(prices, df_12h, h4_12h)
    l1_aligned = align_htf_to_ltf(prices, df_12h, l1_12h)
    l2_aligned = align_htf_to_ltf(prices, df_12h, l2_12h)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    l4_aligned = align_htf_to_ltf(prices, df_12h, l4_12h)
    
    # 12h volume confirmation: current volume > 20-period average
    volume_12h = df_12h['volume'].values
    vol_avg_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure sufficient data
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(h1_aligned[i]) or np.isnan(l1_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 12h volume (aligned)
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        vol_confirm = vol_12h_current > vol_avg_20_aligned[i]
        
        # Price levels for current bar
        h1 = h1_aligned[i]
        h2 = h2_aligned[i]
        h3 = h3_aligned[i]
        h4 = h4_aligned[i]
        l1 = l1_aligned[i]
        l2 = l2_aligned[i]
        l3 = l3_aligned[i]
        l4 = l4_aligned[i]
        
        # Long conditions: bounce from support levels with volume
        long_signal = vol_confirm and (
            (close[i] > l1 and low[i] <= l1) or  # bounce from L1
            (close[i] > l2 and low[i] <= l2) or  # bounce from L2
            (close[i] > l3 and low[i] <= l3) or  # bounce from L3
            (close[i] > l4 and low[i] <= l4)     # bounce from L4
        )
        
        # Short conditions: rejection from resistance levels with volume
        short_signal = vol_confirm and (
            (close[i] < h1 and high[i] >= h1) or  # rejection from H1
            (close[i] < h2 and high[i] >= h2) or  # rejection from H2
            (close[i] < h3 and high[i] >= h3) or  # rejection from H3
            (close[i] < h4 and high[i] >= h4)     # rejection from H4
        )
        
        # Exit conditions: price moves to opposite side of pivot
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals