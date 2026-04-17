#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w High-Low Range for regime detection ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 12-week high-low range
    range_12w = np.full(len(high_1w), np.nan)
    for i in range(len(high_1w)):
        if i >= 11:
            range_12w[i] = np.max(high_1w[i-11:i+1]) - np.min(low_1w[i-11:i+1])
        elif i > 0:
            range_12w[i] = np.max(high_1w[0:i+1]) - np.min(low_1w[0:i+1])
        else:
            range_12w[i] = high_1w[i] - low_1w[i]
    
    # Current weekly range (1-period)
    weekly_range = high_1w - low_1w
    
    # Range ratio: current weekly range / 12-week average range
    range_ratio = np.full(len(high_1w), np.nan)
    for i in range(len(high_1w)):
        if not np.isnan(range_12w[i]) and range_12w[i] > 0:
            range_ratio[i] = weekly_range[i] / range_12w[i]
    
    range_ratio_aligned = align_htf_to_ltf(prices, df_1w, range_ratio)
    
    # === 1d Williams %R for mean reversion ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    highest_high = np.full(len(high_1d), np.nan)
    lowest_low = np.full(len(low_1d), np.nan)
    period = 14
    
    for i in range(len(high_1d)):
        if i >= period - 1:
            highest_high[i] = np.max(high_1d[i-(period-1):i+1])
            lowest_low[i] = np.min(low_1d[i-(period-1):i+1])
        elif i > 0:
            highest_high[i] = np.max(high_1d[0:i+1])
            lowest_low[i] = np.min(low_1d[0:i+1])
        else:
            highest_high[i] = high_1d[i]
            lowest_low[i] = low_1d[i]
    
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    williams_r = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(highest_high[i]) and not np.isnan(lowest_low[i]):
            denominator = highest_high[i] - lowest_low[i]
            if denominator != 0:
                williams_r[i] = -100 * (highest_high[i] - close_1d[i]) / denominator
            else:
                williams_r[i] = -50  # neutral when no range
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # === 12h Volume spike confirmation ===
    # Calculate 24-period average volume (2 days of 12h data)
    vol_ma_24 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 23:
            vol_ma_24[i] = np.mean(volume[i-23:i+1])
        elif i > 0:
            vol_ma_24[i] = np.mean(volume[max(0, i-12):i+1])
        else:
            vol_ma_24[i] = volume[0]
    
    # Volume spike: current volume > 2.0 x 24-period average
    vol_spike = volume > vol_ma_24 * 2.0
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(range_ratio_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume spike
        if position == 0:
            # Long: Williams %R < -80 (oversold) + low volatility regime + volume spike
            if (williams_r_aligned[i] < -80 and 
                range_ratio_aligned[i] < 0.3 and  # low volatility regime
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Williams %R > -20 (overbought) + low volatility regime + volume spike
            elif (williams_r_aligned[i] > -20 and 
                  range_ratio_aligned[i] < 0.3 and  # low volatility regime
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR high volatility regime
            if (williams_r_aligned[i] > -20 or 
                range_ratio_aligned[i] > 0.7):  # high volatility regime
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR high volatility regime
            if (williams_r_aligned[i] < -80 or 
                range_ratio_aligned[i] > 0.7):  # high volatility regime
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_RangeRatio_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0