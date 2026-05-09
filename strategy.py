#!/usr/bin/env python3
# 6h_Stochastic_50_Cross_1dTrend_VolumeFilter
# Hypothesis: 6s Stochastic(14,3,3) crossing above/below 50 with 1d EMA50 trend filter and volume confirmation.
# Long when: 1d trend up (close > EMA50), %K crosses above 50, volume > 1.5x average.
# Short when: 1d trend down (close < EMA50), %K crosses below 50, volume > 1.5x average.
# Uses 6s timeframe for entry timing, 1d for trend filter. Designed for 15-30 trades/year to avoid fee drag.

name = "6h_Stochastic_50_Cross_1dTrend_VolumeFilter"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = (close_1d[i] * 2 + ema50_1d[i-1] * 48) / 50
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Stochastic(14,3,3) on 6h data
    lookback = 14
    k_smooth = 3
    d_smooth = 3
    
    lowest_low = np.full_like(low, np.nan)
    highest_high = np.full_like(high, np.nan)
    
    for i in range(lookback - 1, n):
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
    
    stoch_k_raw = np.full_like(close, np.nan)
    valid_range = (highest_high != lowest_low) & ~np.isnan(highest_high) & ~np.isnan(lowest_low)
    stoch_k_raw[valid_range] = (close[valid_range] - lowest_low[valid_range]) / (highest_high[valid_range] - lowest_low[valid_range]) * 100
    
    # Smooth %K to get fast %K
    stoch_k = np.full_like(close, np.nan)
    if n >= k_smooth:
        for i in range(k_smooth - 1, n):
            start = i - k_smooth + 1
            valid_k = stoch_k_raw[start:i+1]
            valid_k = valid_k[~np.isnan(valid_k)]
            if len(valid_k) > 0:
                stoch_k[i] = np.mean(valid_k)
    
    # Smooth to get %D (slow)
    stoch_d = np.full_like(close, np.nan)
    if n >= d_smooth:
        for i in range(d_smooth - 1, n):
            start = i - d_smooth + 1
            valid_k = stoch_k[start:i+1]
            valid_k = valid_k[~np.isnan(valid_k)]
            if len(valid_k) > 0:
                stoch_d[i] = np.mean(valid_k)
    
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
    
    start_idx = max(lookback, k_smooth, d_smooth, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close[i] > ema50_1d_aligned[i]
        
        if position == 0:
            # Enter long: 1d trend up + %K crosses above 50 + volume confirmation
            if trend_up and stoch_k[i-1] < 50 and stoch_k[i] >= 50 and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: 1d trend down + %K crosses below 50 + volume confirmation
            elif not trend_up and stoch_k[i-1] > 50 and stoch_k[i] <= 50 and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 1d trend turns down or %K crosses below 50
            if not trend_up or (stoch_k[i-1] >= 50 and stoch_k[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 1d trend turns up or %K crosses above 50
            if trend_up or (stoch_k[i-1] <= 50 and stoch_k[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals