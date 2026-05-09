#!/usr/bin/env python3
# 6h_4hEMA200_VolumeSpike_Breakout
# Hypothesis: 6s strategy uses 4h EMA200 as long-term trend filter (avoids counter-trend trades).
# Entry on 6h Donchian breakout (20-period) with volume confirmation (>2x 20-period avg volume).
# Works in bull/bear markets by following higher timeframe trend.
# Target: 20-50 trades/year to minimize fee drag.

name = "6h_4hEMA200_VolumeSpike_Breakout"
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
    
    # Get 4h data for EMA200 trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate 4h EMA200
    ema_200_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 200:
        ema_200_4h[199] = np.mean(close_4h[0:200])
        for i in range(200, len(close_4h)):
            ema_200_4h[i] = (ema_200_4h[i-1] * 199 + close_4h[i]) / 200
    
    # Align 4h EMA200 to 6h timeframe
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Volume spike filter: current volume / 20-period average volume (on 6h)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    # Donchian channel (20-period) on 6h
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    if len(high) >= 20:
        for i in range(19, len(high)):
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # Ensure Donchian and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_4h_aligned[i]) or 
            np.isnan(volume_ratio[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 20-period high AND uptrend (close > 4h EMA200) AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_200_4h_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-period low AND downtrend (close < 4h EMA200) AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_200_4h_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 20-period low OR trend reversal (close < 4h EMA200)
            if close[i] < lowest_low[i] or close[i] < ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 20-period high OR trend reversal (close > 4h EMA200)
            if close[i] > highest_high[i] or close[i] > ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals