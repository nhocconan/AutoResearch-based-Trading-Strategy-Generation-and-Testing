#!/usr/bin/env python3
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
    
    # Get 12h data for Donchian channels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian(20) channels on 12h
    donchian_high = np.full(len(df_12h), np.nan)
    donchian_low = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        donchian_high[i] = np.max(high_12h[i-20:i])
        donchian_low[i] = np.min(low_12h[i-20:i])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate 12h EMA(50) for trend filter
    ema_12h_50 = np.full(len(df_12h), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_12h)):
        if i < 49:
            ema_12h_50[i] = np.mean(close_12h[:i+1]) if i > 0 else close_12h[i]
        else:
            if np.isnan(ema_12h_50[i-1]):
                ema_12h_50[i] = np.mean(close_12h[i-49:i+1])
            else:
                ema_12h_50[i] = close_12h[i] * alpha + ema_12h_50[i-1] * (1 - alpha)
    
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Get daily data for volume spike detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate volume ratio: current 12h volume / average of last 20 days
    # First, we need to aggregate 12h volume to daily equivalent
    # Since we don't have direct aggregation, we'll use 1h volume as proxy for intraday
    # But simpler: use current 4h volume vs 20-period average of 4h volume
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    volume_ratio = np.full(n, np.nan)
    valid_vol = (~np.isnan(vol_ma_20)) & (vol_ma_20 > 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma_20[valid_vol]
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_12h_50_aligned[i]) or
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume spike: current volume > 1.5x 20-period average
        vol_spike = volume_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume spike + price above EMA(50)
            if (price > donchian_high_aligned[i] and 
                vol_spike and 
                price > ema_12h_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume spike + price below EMA(50)
            elif (price < donchian_low_aligned[i] and 
                  vol_spike and 
                  price < ema_12h_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below EMA(50) or Donchian low
            if (price < ema_12h_50_aligned[i] or 
                price < donchian_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above EMA(50) or Donchian high
            if (price > ema_12h_50_aligned[i] or 
                price > donchian_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_VolumeSpike_12hEMA50_v1"
timeframe = "4h"
leverage = 1.0