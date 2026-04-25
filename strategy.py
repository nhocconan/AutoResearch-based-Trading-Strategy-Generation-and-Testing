#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike Confirmation
Hypothesis: Donchian channel breakouts capture momentum. Weekly pivot (from prior week) provides
institutional bias: only long when price above weekly pivot, short when below. Volume spike
confirms institutional participation. Designed for 6h timeframe to target 50-150 total trades
over 4 years. Works in bull markets via upside breakouts above weekly pivot and in bear markets
via downside breakdowns below weekly pivot. Uses discrete position sizing (0.25) to limit fee drag.
"""

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
    
    # Get 1w data for weekly pivot (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week: P = (H+L+C)/3
    weekly_high = df_1w['high'].shift(1).values
    weekly_low = df_1w['low'].shift(1).values
    weekly_close = df_1w['close'].shift(1).values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (no extra delay - pivot known at week open)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian(20) channels on 6h data
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-19:i+1])
        donchian_l[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, volume MA, and weekly pivot
    start_idx = max(20, 20)  # 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_h[i]) or 
            np.isnan(donchian_l[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high = donchian_h[i]
        donchian_low = donchian_l[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average (tighter for fewer trades)
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Breakout conditions
        breakout_above = curr_close > donchian_high
        breakout_below = curr_close < donchian_low
        
        # Weekly pivot filter: price relative to weekly pivot
        price_above_pivot = curr_close > weekly_pivot_val
        price_below_pivot = curr_close < weekly_pivot_val
        
        if position == 0:
            # Long: break above Donchian high + price above weekly pivot + volume confirmation
            long_signal = breakout_above and price_above_pivot and volume_confirm
            # Short: break below Donchian low + price below weekly pivot + volume confirmation
            short_signal = breakout_below and price_below_pivot and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Donchian low OR price crosses below weekly pivot
            if curr_close < donchian_low or curr_close < weekly_pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above Donchian high OR price crosses above weekly pivot
            if curr_close > donchian_high or curr_close > weekly_pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0