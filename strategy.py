#!/usr/bin/env python3
# 1h_FisherTransform_4hTrend_1dVolumeFilter
# Hypothesis: Ehlers Fisher Transform on 1h for reversal signals, filtered by 4h EMA50 trend and 1d volume spike.
# Works in bull/bear: Trend filter ensures trades align with higher timeframe direction, volume confirms institutional participation.
# Fisher Transform identifies extreme price movements likely to reverse, providing precise entry/exit timing.
# Uses EMA for smooth trend and volume ratio for confirmation to reduce false signals.

name = "1h_FisherTransform_4hTrend_1dVolumeFilter"
timeframe = "1h"
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
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[0:50])
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = (ema_50_4h[i-1] * 49 + close_4h[i]) / 50
    
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d volume ratio (current / 20-period average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        vol_ma_1d[19] = np.mean(volume_1d[0:20])
        for i in range(20, len(volume_1d)):
            vol_ma_1d[i] = (vol_ma_1d[i-1] * 19 + volume_1d[i]) / 20
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma_1d_aligned)) & (vol_ma_1d_aligned != 0)
    volume_ratio[valid] = volume[valid] / vol_ma_1d_aligned[valid]
    
    # Calculate Ehlers Fisher Transform on 1h prices
    # Normalize price to [-1, 1] range over lookback period
    lookback = 10
    hl2 = (high + low) / 2
    max_hl2 = np.full_like(hl2, np.nan)
    min_hl2 = np.full_like(hl2, np.nan)
    
    for i in range(lookback-1, n):
        max_hl2[i] = np.max(hl2[i-lookback+1:i+1])
        min_hl2[i] = np.min(hl2[i-lookback+1:i+1])
    
    # Avoid division by zero
    diff = max_hl2 - min_hl2
    diff[diff == 0] = 1e-10
    
    # Normalize to [-1, 1]
    value = 2 * ((hl2 - min_hl2) / diff) - 1
    # Clamp to [-0.999, 0.999] to prevent infinity in Fisher transform
    value = np.clip(value, -0.999, 0.999)
    
    # Fisher Transform
    fish = np.full_like(value, np.nan)
    for i in range(1, n):
        fish[i] = 0.5 * np.log((1 + value[i]) / (1 - value[i])) + 0.5 * fish[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 50)  # Ensure Fisher and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(fish[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Fisher crosses above -1.5 (oversold reversal) AND uptrend AND volume spike
            if (fish[i] > -1.5 and fish[i-1] <= -1.5 and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_ratio[i] > 1.8):
                signals[i] = 0.20
                position = 1
            # Enter short: Fisher crosses below +1.5 (overbought reversal) AND downtrend AND volume spike
            elif (fish[i] < 1.5 and fish[i-1] >= 1.5 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_ratio[i] > 1.8):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Fisher crosses below +0.5 (profit take) OR trend reversal
            if (fish[i] < 0.5 and fish[i-1] >= 0.5) or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Fisher crosses above -0.5 (profit take) OR trend reversal
            if (fish[i] > -0.5 and fish[i-1] <= -0.5) or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals