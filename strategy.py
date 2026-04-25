#!/usr/bin/env python3
"""
6h Donchian(20) Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: Donchian channel breakouts capture momentum moves. Weekly pivot (R1/S1) from 1w timeframe provides 
institutional bias - only take breakouts aligned with weekly trend. Volume confirmation ensures participation. 
Designed for low trade frequency (12-37/year) on 6h timeframe to work in both bull and bear markets via 
trend following with institutional alignment.
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
    
    # Get 1w data for weekly pivot (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivots (R1, S1) from previous week's data
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Weekly Camarilla R1/S1: close ± (high-low)*1.1/2
    weekly_r1 = prev_week_close + (prev_week_high - prev_week_low) * 1.1 / 2
    weekly_s1 = prev_week_close - (prev_week_high - prev_week_low) * 1.1 / 2
    
    # Align to LTF (6h) - no extra delay needed as pivots based on completed weekly bar
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Get 1d data for additional trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channel (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, volume MA, EMA, and weekly pivot alignment
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper_donchian = donchian_high[i]
        lower_donchian = donchian_low[i]
        ema_trend = ema_34_1d_aligned[i]
        weekly_r1 = weekly_r1_aligned[i]
        weekly_s1 = weekly_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND volume spike AND price > weekly R1 (bullish bias) AND price > 1d EMA34
            long_entry = (curr_close > upper_donchian) and vol_spike and (curr_close > weekly_r1) and (curr_close > ema_trend)
            # Short: price breaks below Donchian low AND volume spike AND price < weekly S1 (bearish bias) AND price < 1d EMA34
            short_entry = (curr_close < lower_donchian) and vol_spike and (curr_close < weekly_s1) and (curr_close < ema_trend)
            
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
            # Exit: price crosses below Donchian low (breakdown) OR price crosses below weekly S1 (bias change) OR price crosses below EMA (trend change)
            if (curr_close < lower_donchian) or (curr_close < weekly_s1) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Donchian high (breakout) OR price crosses above weekly R1 (bias change) OR price crosses above EMA (trend change)
            if (curr_close > upper_donchian) or (curr_close > weekly_r1) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotR1S1_VolumeSpike_1dEMA34"
timeframe = "6h"
leverage = 1.0