# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h trend following using 1-week price position within weekly range.
In bull markets, price tends to stay in upper half of weekly range; in bear markets,
price tends to stay in lower half. We go long when price is in upper 40% of weekly
range with momentum confirmation, short when in lower 40%. Uses volume filter to
avoid chop. Target: 15-30 trades/year per symbol.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly high-low range for the past 20 weeks
    highest_high_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    lowest_low_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    weekly_range = highest_high_20w - lowest_low_20w
    
    # Calculate where current weekly close sits within the 20-week range (0-1)
    # Avoid division by zero
    range_position = np.where(weekly_range > 0, 
                              (df_1w['close'].values - lowest_low_20w) / weekly_range, 
                              0.5)
    
    # Weekly momentum: rate of change over 4 weeks
    roc_4w = pd.Series(df_1w['close']).pct_change(periods=4).values
    
    # Volume filter: 50-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(vol_ma[i]) or np.isnan(range_position[-1]) if len(range_position) == 0 else False:
            continue
            
        # Get weekly index for current 6h bar (approx: 6h = 1/28 * 1w)
        idx_1w = i // 28
        if idx_1w < 20:  # Need sufficient weekly data for calculations
            continue
        
        # Previous values to avoid look-ahead
        range_pos_prev = range_position[idx_1w-1] if idx_1w-1 < len(range_position) else range_position[-1]
        roc_prev = roc_4w[idx_1w-1] if idx_1w-1 < len(roc_4w) else roc_4w[-1]
        
        if np.isnan(range_pos_prev) or np.isnan(roc_prev):
            continue
        
        # Create arrays for alignment (using previous values to avoid look-ahead)
        range_arr = np.full(len(df_1w), range_pos_prev)
        roc_arr = np.full(len(df_1w), roc_prev)
        
        # Align to 6h timeframe
        range_pos_6h = align_htf_to_ltf(prices, df_1w, range_arr)[i]
        roc_6h = align_htf_to_ltf(prices, df_1w, roc_arr)[i]
        
        if position == 0:
            # Long: price in upper 40% of weekly range + positive weekly momentum + volume confirmation
            if (range_pos_6h > 0.6 and 
                roc_6h > 0 and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: price in lower 40% of weekly range + negative weekly momentum + volume confirmation
            elif (range_pos_6h < 0.4 and 
                  roc_6h < 0 and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price returns to middle 50% of weekly range or momentum turns negative
            if range_pos_6h < 0.5 or roc_6h < 0:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price returns to middle 50% of weekly range or momentum turns positive
            if range_pos_6h > 0.5 or roc_6h > 0:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_1w_RangePosition_Momentum"
timeframe = "6h"
leverage = 1.0