#!/usr/bin/env python3
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
    
    # Get daily data for Donchian breakout and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Donchian channel (20-period)
    highest_high_20 = np.full(len(df_1d), np.nan)
    lowest_low_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        highest_high_20[i] = np.max(high_1d[i-19:i+1])
        lowest_low_20[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate average daily volume (20-period)
    avg_volume_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        avg_volume_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align Donchian levels and average volume to 6h timeframe
    highest_high_20_aligned = align_htf_to_ltf(prices, df_1d, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    avg_volume_20_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20)
    
    # Get weekly data for trend filter: EMA(50) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_50 = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (50 + 1)
    for i in range(len(close_1w)):
        if i < 49:
            ema_1w_50[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_1w_50[i-1]):
                ema_1w_50[i] = np.mean(close_1w[i-49:i+1])
            else:
                ema_1w_50[i] = close_1w[i] * alpha_w + ema_1w_50[i-1] * (1 - alpha_w)
    
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20_aligned[i]) or 
            np.isnan(lowest_low_20_aligned[i]) or
            np.isnan(avg_volume_20_aligned[i]) or
            np.isnan(ema_1w_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume_20_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average daily volume
        volume_confirm = vol > 1.5 * avg_vol if avg_vol > 0 else False
        
        if position == 0:
            # Long breakout: price breaks above 20-day high + volume + weekly uptrend
            if (price > highest_high_20_aligned[i] and 
                volume_confirm and 
                ema_1w_50_aligned[i] > ema_1w_50_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below 20-day low + volume + weekly downtrend
            elif (price < lowest_low_20_aligned[i] and 
                  volume_confirm and 
                  ema_1w_50_aligned[i] < ema_1w_50_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to midpoint of Donchian channel or weekly trend turns down
            midpoint = (highest_high_20_aligned[i] + lowest_low_20_aligned[i]) / 2
            if (price < midpoint or 
                ema_1w_50_aligned[i] < ema_1w_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to midpoint of Donchian channel or weekly trend turns up
            midpoint = (highest_high_20_aligned[i] + lowest_low_20_aligned[i]) / 2
            if (price > midpoint or 
                ema_1w_50_aligned[i] > ema_1w_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DonchianBreakout_WeeklyEMA50_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0