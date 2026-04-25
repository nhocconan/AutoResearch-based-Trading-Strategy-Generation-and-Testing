#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
Hypothesis: Donchian breakouts capture strong momentum. Weekly pivot (from 1w) provides institutional bias:
- Price above weekly pivot = bullish bias (favor longs)
- Price below weekly pivot = bearish bias (favor shorts)
Volume spike confirms participation. Designed for 6h timeframe to achieve 12-37 trades/year per symbol,
minimizing fee drag while working in both bull and bear markets via directional filtering.
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
    
    # Get 6h data for Donchian channels (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 6h high/low
    donchian_high = pd.Series(df_6h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_6h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (same as primary, so no shift needed but use helper for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Get weekly data for pivot point calculation (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot point: (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe with extra delay for weekly bar confirmation
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot, additional_delay_bars=1)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20), volume MA, and weekly pivot
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals with weekly pivot bias
            # Long: price breaks above Donchian high AND volume spike AND price > weekly pivot (bullish bias)
            long_entry = (curr_close > donchian_high_aligned[i]) and vol_spike and (curr_close > weekly_pivot_val)
            # Short: price breaks below Donchian low AND volume spike AND price < weekly pivot (bearish bias)
            short_entry = (curr_close < donchian_low_aligned[i]) and vol_spike and (curr_close < weekly_pivot_val)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below Donchian low OR price crosses below weekly pivot (bias change)
            if (curr_close < donchian_low_aligned[i]) or (curr_close < weekly_pivot_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Donchian high OR price crosses above weekly pivot (bias change)
            if (curr_close > donchian_high_aligned[i]) or (curr_close > weekly_pivot_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0