#!/usr/bin/env python3
# 6h_Donchian20_Breakout_12hTrend_VolumeSpike
# Hypothesis: Buy breakouts above 6h Donchian high(20) when 12h EMA50 is rising and volume >2x 20-bar average; sell breakdowns below Donchian low(20) when 12h EMA50 is falling and volume >2x average. Uses volume spike to confirm conviction and 12h EMA50 slope for trend filter. Designed for 50-150 total trades over 4 years on 6h timeframe.

name = "6h_Donchian20_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50)
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * 2 + ema_50_12h[i-1] * 48) / 50
    
    # Calculate 12h EMA(50) slope (rising/falling)
    ema_slope_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 51:
        ema_slope_12h[50:] = ema_50_12h[50:] - ema_50_12h[49:-1]
    
    # Align 12h EMA and slope to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_slope_12h)
    
    # Calculate Donchian channels (20-period) on 6h data
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    if len(high) >= 20:
        for i in range(19, len(high)):
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: 6h volume / 20-period average volume
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
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_slope_12h_aligned[i]) or \
           np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Breakout above Donchian high + rising 12h EMA50 + volume spike
            if close[i] > donchian_high[i] and ema_slope_12h_aligned[i] > 0 and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: Breakdown below Donchian low + falling 12h EMA50 + volume spike
            elif close[i] < donchian_low[i] and ema_slope_12h_aligned[i] < 0 and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Breakdown below Donchian low or 12h EMA50 turns falling
            if close[i] < donchian_low[i] or ema_slope_12h_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Breakout above Donchian high or 12h EMA50 turns rising
            if close[i] > donchian_high[i] or ema_slope_12h_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals