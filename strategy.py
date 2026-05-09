#!/usr/bin/env python3
# 1d_Donchian_Breakout_WeeklyTrend_Volume
# Hypothesis: Uses weekly trend filter with daily Donchian(20) breakout on 1d timeframe.
# Long when weekly trend up and price breaks above Donchian upper band with volume confirmation.
# Short when weekly trend down and price breaks below Donchian lower band with volume confirmation.
# Weekly trend determined by price above/below weekly EMA34.
# Target: 15-30 trades/year per symbol with disciplined risk management.

name = "1d_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema34_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 34:
        ema34_weekly[33] = np.mean(close_weekly[0:34])
        for i in range(34, len(close_weekly)):
            ema34_weekly[i] = (close_weekly[i] * 2 + ema34_weekly[i-1] * 32) / 34
    
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Get daily data for Donchian channels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Calculate Donchian channels (20-period)
    donchian_upper = np.full_like(high_daily, np.nan)
    donchian_lower = np.full_like(low_daily, np.nan)
    
    for i in range(len(df_daily)):
        if i < 19:
            continue
        donchian_upper[i] = np.max(high_daily[i-19:i+1])
        donchian_lower[i] = np.min(low_daily[i-19:i+1])
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_daily, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_daily, donchian_lower)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 1)  # Need weekly EMA, daily Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_weekly_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        weekly_up = close[i] > ema34_weekly_aligned[i]
        
        if position == 0:
            # Enter long: weekly trend up + price breaks above Donchian upper + volume confirmation
            if weekly_up and close[i] > donchian_upper_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly trend down + price breaks below Donchian lower + volume confirmation
            elif not weekly_up and close[i] < donchian_lower_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or price breaks below Donchian lower
            if not weekly_up or close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or price breaks above Donchian upper
            if weekly_up or close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals