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
    
    # Get weekly data for 200 EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_200 = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (200 + 1)
    for i in range(len(close_1w)):
        if i < 199:
            ema_1w_200[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_1w_200[i-1]):
                ema_1w_200[i] = np.mean(close_1w[i-199:i+1])
            else:
                ema_1w_200[i] = close_1w[i] * alpha_w + ema_1w_200[i-1] * (1 - alpha_w)
    
    ema_1w_200_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_200)
    
    # Get daily data for ATR(14) and Donchian(20)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Donchian channels (20-period)
    highest_20 = np.full(len(df_1d), np.nan)
    lowest_20 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:
            highest_20[i] = np.max(high_1d[i-19:i+1])
            lowest_20[i] = np.min(low_1d[i-19:i+1])
        else:
            highest_20[i] = np.max(high_1d[:i+1])
            lowest_20[i] = np.min(low_1d[:i+1])
    
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Calculate daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_20 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:
            vol_avg_20[i] = np.mean(vol_1d[i-19:i+1])
        else:
            vol_avg_20[i] = np.mean(vol_1d[:i+1])
    
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_200_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(highest_20_aligned[i]) or
            np.isnan(lowest_20_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_14_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x daily average volume
        vol_ma = vol_avg_20_aligned[i]
        volume_filter = volume[i] > 1.5 * vol_ma if vol_ma > 0 else False
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend + volume
            if (price > highest_20_aligned[i] and 
                ema_1w_200_aligned[i] > ema_1w_200_aligned[i-1] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + weekly downtrend + volume
            elif (price < lowest_20_aligned[i] and 
                  ema_1w_200_aligned[i] < ema_1w_200_aligned[i-1] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or weekly trend turns down
            if (price < lowest_20_aligned[i] or 
                ema_1w_200_aligned[i] < ema_1w_200_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or weekly trend turns up
            if (price > highest_20_aligned[i] or 
                ema_1w_200_aligned[i] > ema_1w_200_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DonchianBreakout_WeeklyEMA200_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0