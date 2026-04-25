#!/usr/bin/env python3
"""
6h Donchian(20) breakout + weekly pivot direction + volume confirmation
Hypothesis: Donchian breakouts capture strong momentum, while weekly pivot direction
provides institutional bias. Volume confirmation filters false breakouts. Designed
for 6h timeframe targeting 12-37 trades/year. Works in bull via breakout continuation
and in bear via mean-reversion when price rejects weekly pivot levels. Uses proper
MTF loading with get_htf_data called once before loop.
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
    
    # Get weekly data for pivot direction (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get 1d data for volume MA (more stable than 6h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on 1d
    vol_ma_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Donchian channels (20-period) on 6h
    if len(close) >= 20:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        for i in range(20-1, n):
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for weekly pivot, Donchian, and volume MA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Volume spike: current 6h volume > 1.5 * 20-period 1d volume MA (scaled)
        # Approximate 6h volume vs daily: 6h is 1/4 of day, so scale accordingly
        volume_spike = curr_volume > 1.5 * (vol_ma / 4.0)  # vol_ma is daily, divide by 4 for 6h equivalent
        
        if position == 0:
            # Long: price breaks above upper Donchian AND above weekly pivot AND volume spike
            long_condition = (curr_close > upper_channel) and (curr_close > weekly_pivot_val) and volume_spike
            # Short: price breaks below lower Donchian AND below weekly pivot AND volume spike
            short_condition = (curr_close < lower_channel) and (curr_close < weekly_pivot_val) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price breaks below lower Donchian (reversal) or drops below weekly pivot
            if curr_close < lower_channel or curr_close < weekly_pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian (reversal) or rises above weekly pivot
            if curr_close > upper_channel or curr_close > weekly_pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0