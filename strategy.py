#!/usr/bin/env python3
# 6h_Donchian_WeeklyPivot_Filter_Volume
# Hypothesis: Uses 6-hour Donchian channel breakouts filtered by weekly pivot direction and volume confirmation.
# Enters long when price breaks above Donchian(20) high with weekly pivot above prior week close and volume spike.
# Enters short when price breaks below Donchian(20) low with weekly pivot below prior week close and volume spike.
# Exits when price returns to Donchian midpoint or weekly pivot flips direction.
# Weekly pivot acts as regime filter to avoid counter-trend trades in both bull and bear markets.
# Targets 12-37 trades per year on 6h timeframe with position size 0.25.

name = "6h_Donchian_WeeklyPivot_Filter_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L
    # S1 = 2*PP - H
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_high[0] = 0
    prev_week_low[0] = 0
    prev_week_close[0] = 0
    
    weekly_pp = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_r1 = 2 * weekly_pp - prev_week_low
    weekly_s1 = 2 * weekly_pp - prev_week_high
    
    # Align weekly pivot levels to 6h
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Determine weekly trend: above PP = uptrend, below PP = downtrend
    weekly_uptrend = weekly_pp_aligned > np.roll(weekly_pp_aligned, 1)
    weekly_downtrend = weekly_pp_aligned < np.roll(weekly_pp_aligned, 1)
    
    # Get daily data for volume confirmation (more responsive than weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Align daily OHLC to 6h for Donchian calculation
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, df_1d['high'].values)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, df_1d['low'].values)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    # Calculate Donchian channel (20-period) using daily data aligned to 6h
    donchian_high = pd.Series(daily_high_aligned).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(daily_low_aligned).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: current volume > 2.0 * 20-period average (using 6h volume)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(weekly_pp_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high with weekly uptrend and volume spike
            if (close[i] > donchian_high[i] and 
                weekly_pp_aligned[i] > np.roll(weekly_pp_aligned, 1)[i] and  # weekly PP rising
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with weekly downtrend and volume spike
            elif (close[i] < donchian_low[i] and 
                  weekly_pp_aligned[i] < np.roll(weekly_pp_aligned, 1)[i] and  # weekly PP falling
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to Donchian midpoint or weekly trend turns down
            if (close[i] <= donchian_mid[i] or 
                weekly_pp_aligned[i] < np.roll(weekly_pp_aligned, 1)[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian midpoint or weekly trend turns up
            if (close[i] >= donchian_mid[i] or 
                weekly_pp_aligned[i] > np.roll(weekly_pp_aligned, 1)[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals