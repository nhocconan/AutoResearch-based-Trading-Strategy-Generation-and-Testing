# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w price position and 1d volume spike
# Breakouts above/below 20-period high/low indicate strong momentum.
# 1w price position (>0.8 or <0.2) filters for extreme momentum in weekly trend.
# 1d volume spike confirms institutional participation.
# This combination works in both bull and bear markets by requiring extreme weekly positioning.
# Targets 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.

name = "6h_Donchian20_1wPosition_1dVolume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for price position (0-1 range within weekly range)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w price position: where current price is within weekly range
    highest_high_1w = df_1w['high'].values
    lowest_low_1w = df_1w['low'].values
    weekly_range = highest_high_1w - lowest_low_1w
    # Avoid division by zero
    weekly_range = np.where(weekly_range == 0, 1, weekly_range)
    price_position_1w = (df_1w['close'].values - lowest_low_1w) / weekly_range
    
    # Extreme positions: >0.8 (near weekly high) or <0.2 (near weekly low)
    extreme_high_1w = price_position_1w > 0.8
    extreme_low_1w = price_position_1w < 0.2
    extreme_high_6h = align_htf_to_ltf(prices, df_1w, extreme_high_1w)
    extreme_low_6h = align_htf_to_ltf(prices, df_1w, extreme_low_1w)
    
    # Get 1d data for volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d volume spike detection
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean()
    vol_spike_1d = df_1d['volume'].values > (vol_ma.values * 2.0)
    vol_spike = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Donchian(20) on 6h
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback, len(high)):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Align Donchian levels (use previous bar's values to avoid look-ahead)
    highest_high_aligned = np.roll(highest_high, 1)
    lowest_low_aligned = np.roll(lowest_low, 1)
    highest_high_aligned[0] = np.nan
    lowest_low_aligned[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback + 1, 20)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(extreme_high_6h[i]) or np.isnan(extreme_low_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 20-period high, extreme weekly high, volume spike
            if close[i] > highest_high_aligned[i] and extreme_high_6h[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-period low, extreme weekly low, volume spike
            elif close[i] < lowest_low_aligned[i] and extreme_low_6h[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 20-period low or weekly extreme fades
            if close[i] < lowest_low_aligned[i] or not extreme_high_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 20-period high or weekly extreme fades
            if close[i] > highest_high_aligned[i] or not extreme_low_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals