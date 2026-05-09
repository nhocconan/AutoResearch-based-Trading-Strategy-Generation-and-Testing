#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_VolumeFilter
# Hypothesis: In both bull and bear markets, price tends to respect weekly pivot levels.
# Long when price breaks above Donchian(20) high AND closes above weekly pivot (S1) with volume spike.
# Short when price breaks below Donchian(20) low AND closes below weekly pivot (R1) with volume spike.
# Uses weekly pivot for structural bias and volume to confirm breakout strength.
# Designed for 6h timeframe with low trade frequency to avoid fee drag.

name = "6h_WeeklyPivot_DonchianBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly high, low, close from prior completed week
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    # Pivot point = (H + L + C)/3
    pp = (wh + wl + wc) / 3.0
    # Support 1 = (2 * PP) - High
    s1 = (2 * pp) - wh
    # Resistance 1 = (2 * PP) - Low
    r1 = (2 * pp) - wl
    
    # Align weekly pivot levels to 6h timeframe (using prior week's values)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # Donchian channel (20-period) on 6h
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback - 1, len(high)):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume average (20-period) for spike detection
    vol_ma = np.full_like(volume, np.nan)
    for i in range(lookback - 1, len(volume)):
        vol_ma[i] = np.mean(volume[i - lookback + 1:i + 1])
    vol_ratio = volume / vol_ma  # Current volume relative to average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback - 1, 20)  # Ensure Donchian and volume ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian high AND close > weekly S1 AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > s1_aligned[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < Donchian low AND close < weekly R1 AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < r1_aligned[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below weekly pivot (S1) or Donchian low
            if close[i] < s1_aligned[i] or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above weekly pivot (R1) or Donchian high
            if close[i] > r1_aligned[i] or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals