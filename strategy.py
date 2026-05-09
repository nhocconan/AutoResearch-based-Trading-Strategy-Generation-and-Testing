#!/usr/bin/env python3
# 6h_DonchianBreakout_WeeklyPivotDirection_Volume
# Hypothesis: 6h Donchian(20) breakout with weekly pivot (WPP) direction filter and volume confirmation.
# Long when weekly trend up (price > WPP) and price breaks above Donchian upper band with volume > 1.5x avg.
# Short when weekly trend down (price < WPP) and price breaks below Donchian lower band with volume > 1.5x avg.
# Weekly pivot provides structural bias to avoid counter-trend trades, reducing whipsaw in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

name = "6h_DonchianBreakout_WeeklyPivotDirection_Volume"
timeframe = "6h"
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
    
    # Get weekly data for pivot calculation and trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate Weekly Pivot Point (WPP) = (H + L + C) / 3
    wpp = (high_w + low_w + close_w) / 3.0
    
    # Get daily data for Donchian calculation (more stable than intraday for pivot-based systems)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    
    # Calculate Donchian channels (20-period)
    upper_d = np.full_like(high_d, np.nan)
    lower_d = np.full_like(low_d, np.nan)
    
    for i in range(19, len(high_d)):
        upper_d[i] = np.max(high_d[i-19:i+1])
        lower_d[i] = np.min(low_d[i-19:i+1])
    
    # Align weekly and daily indicators to 6h timeframe
    wpp_aligned = align_htf_to_ltf(prices, df_w, wpp)
    upper_d_aligned = align_htf_to_ltf(prices, df_d, upper_d)
    lower_d_aligned = align_htf_to_ltf(prices, df_d, lower_d)
    
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
    
    start_idx = max(19, 1, 20)  # Need Donchian, WPP, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(wpp_aligned[i]) or np.isnan(upper_d_aligned[i]) or 
            np.isnan(lower_d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend: price above/below WPP
        weekly_up = close[i] > wpp_aligned[i]
        
        if position == 0:
            # Enter long: weekly trend up + price breaks above Donchian upper + volume confirmation
            if weekly_up and close[i] > upper_d_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly trend down + price breaks below Donchian lower + volume confirmation
            elif not weekly_up and close[i] < lower_d_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or price breaks below Donchian lower
            if not weekly_up or close[i] < lower_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or price breaks above Donchian upper
            if weekly_up or close[i] > upper_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals