#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_TrendFilter
Hypothesis: 6h timeframe strategy using weekly pivot levels (from 1w) for trend direction,
Donchian(20) breakout for entry timing, and volume confirmation. Uses 1w trend filter to
avoid counter-trend trades. Designed for low trade frequency (<40/year) to minimize fee drag
while capturing major trend moves in both bull and bear markets.
"""
name = "6h_WeeklyPivot_DonchianBreakout_TrendFilter"
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for weekly pivot calculation
    ph = np.concatenate([[high_1w[0]], high_1w[:-1]])  # previous high
    pl = np.concatenate([[low_1w[0]], low_1w[:-1]])   # previous low
    pc = np.concatenate([[close_1w[0]], close_1w[:-1]]) # previous close
    
    # Calculate weekly pivot points (standard formula)
    pp = (ph + pl + pc) / 3.0                    # Pivot Point
    r1 = 2 * pp - pl                             # Resistance 1
    s1 = 2 * pp - ph                             # Support 1
    r2 = pp + (ph - pl)                          # Resistance 2
    s2 = pp - (ph - pl)                          # Support 2
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate 1w EMA40 for trend filter (more responsive than longer periods)
    ema_40_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 40:
        ema_40_1w[39] = np.mean(close_1w[0:40])
        for i in range(40, len(close_1w)):
            ema_40_1w[i] = (ema_40_1w[i-1] * 39 + close_1w[i]) / 40
    
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Donchian channel (20-period) for breakout signals
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    if len(high) >= 20:
        for i in range(20-1, len(high)):
            donchian_high[i] = np.max(high[i-20+1:i+1])
            donchian_low[i] = np.min(low[i-20+1:i+1])
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 40)  # Ensure Donchian and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_40_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND above weekly pivot (uptrend bias) AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > pp_aligned[i] and 
                close[i] > ema_40_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND below weekly pivot (downtrend bias) AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < pp_aligned[i] and 
                  close[i] < ema_40_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly support OR trend reversal (price < weekly EMA)
            if close[i] < s1_aligned[i] or close[i] < ema_40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly resistance OR trend reversal (price > weekly EMA)
            if close[i] > r1_aligned[i] or close[i] > ema_40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals