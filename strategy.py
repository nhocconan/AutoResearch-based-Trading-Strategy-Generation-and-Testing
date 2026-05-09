#!/usr/bin/env python3
# 6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation
# Hypothesis: Donchian(20) breakout on 6h with weekly pivot direction (from 1w) as trend filter and volume spike confirmation.
# Weekly pivot provides multi-week trend bias, reducing counter-trend trades. Volume surge confirms institutional participation.
# Works in bull/bear: trend filter avoids counter-trend trades, volume confirms breakout strength.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate weekly pivot points (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    wo = df_1w['open'].values
    
    # Previous week values
    pwh = np.concatenate([[wh[0]], wh[:-1]])  # previous week high
    pwl = np.concatenate([[wl[0]], wl[:-1]])  # previous week low
    pwc = np.concatenate([[wc[0]], wc[:-1]])  # previous week close
    
    # Weekly pivot point and support/resistance levels
    pivot = (pwh + pwl + pwc) / 3.0
    r1 = 2 * pivot - pwl
    s1 = 2 * pivot - pwh
    r2 = pivot + (pwh - pwl)
    s2 = pivot - (pwh - pwl)
    r3 = pwh + 2 * (pivot - pwl)
    s3 = pwl - 2 * (pwh - pivot)
    
    # Use weekly pivot direction: price above pivot = bullish bias, below = bearish bias
    weekly_bias = pivot  # simple bias: above pivot = bullish
    
    # Align weekly pivot to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Calculate Donchian(20) channels on 6h
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    if len(high) >= lookback:
        for i in range(lookback - 1, len(high)):
            highest_high[i] = np.max(high[i - lookback + 1:i + 1])
            lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma[i] = np.mean(volume[i - 19:i + 1])
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian(20) and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_bias_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND above weekly pivot (bullish bias) AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > weekly_bias_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND below weekly pivot (bearish bias) AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < weekly_bias_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR below weekly pivot
            if close[i] < lowest_low[i] or close[i] < weekly_bias_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR above weekly pivot
            if close[i] > highest_high[i] or close[i] > weekly_bias_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals