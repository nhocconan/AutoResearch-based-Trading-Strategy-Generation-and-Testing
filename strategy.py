#!/usr/bin/env python3
"""
1d Ehlers Fisher Transform + Volume Spike + Weekly Trend Filter.
Long when Fisher crosses above -1.5 with volume spike and weekly uptrend.
Short when Fisher crosses below +1.5 with volume spike and weekly downtrend.
Exit when Fisher crosses back through zero.
Designed to capture reversals in both bull and bear markets with low trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Ehlers Fisher Transform (9-period)
    # Step 1: Normalize price over period
    period = 9
    hl2 = (high + low) / 2.0
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high[i] = np.max(high[i - period + 1:i + 1])
        lowest_low[i] = np.min(low[i - period + 1:i + 1])
    
    # Avoid division by zero
    diff = highest_high - lowest_low
    diff[diff == 0] = 1e-10
    
    # Normalize to [-1, 1]
    value1 = 2.0 * ((hl2 - lowest_low) / diff - 0.5)
    value1 = np.clip(value1, -0.999, 0.999)  # Prevent extreme values
    
    # Smooth with exponential moving average
    alpha = 0.5
    value2 = np.full(n, np.nan)
    for i in range(n):
        if i == 0:
            value2[i] = value1[i]
        elif np.isnan(value2[i-1]):
            value2[i] = value1[i]
        else:
            value2[i] = alpha * value1[i] + (1 - alpha) * value2[i-1]
    
    # Fisher transform
    fish = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(value2[i]):
            fish[i] = np.nan
        else:
            fish[i] = 0.5 * np.log((1 + value2[i]) / (1 - value2[i]))
    
    # Get weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    alpha_ew = 2.0 / (34 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_1w[i] = close_1w[i]
        elif np.isnan(ema_1w[i-1]):
            ema_1w[i] = close_1w[i]
        else:
            ema_1w[i] = alpha_ew * close_1w[i] + (1 - alpha_ew) * ema_1w[i-1]
    
    # Align weekly EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: volume > 2.0x average (to avoid false signals)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Fisher (9) + volume MA (20) + weekly EMA (34)
    start_idx = max(9, 19, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(fish[i]) or np.isnan(fish[i-1]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        fish_now = fish[i]
        fish_prev = fish[i-1]
        vol_now = volume[i]
        trend_weekly = ema_1w_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Fisher crosses above -1.5 with volume spike and weekly uptrend
            if fish_prev <= -1.5 and fish_now > -1.5 and vol_filter and trend_weekly > close[i]:
                signals[i] = size
                position = 1
            # Short: Fisher crosses below +1.5 with volume spike and weekly downtrend
            elif fish_prev >= 1.5 and fish_now < 1.5 and vol_filter and trend_weekly < close[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Fisher crosses below zero
            if fish_prev > 0 and fish_now <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Fisher crosses above zero
            if fish_prev < 0 and fish_now >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_EhlerFisher_VolumeSpike_WeeklyTrend"
timeframe = "1d"
leverage = 1.0