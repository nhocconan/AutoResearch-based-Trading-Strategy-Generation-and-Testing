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
    
    # Get 12h data for trend filter: EMA(50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h_50 = np.full(len(df_12h), np.nan)
    for i in range(len(close_12h)):
        if i < 49:
            ema_12h_50[i] = np.mean(close_12h[:i+1]) if i > 0 else close_12h[i]
        else:
            ema_12h_50[i] = np.mean(close_12h[i-49:i+1])
    
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Get daily data for Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian high and low
    donch_high = np.full(len(df_1d), np.nan)
    donch_low = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i < 19:
            if i > 0:
                donch_high[i] = np.max(high_1d[:i+1])
                donch_low[i] = np.min(low_1d[:i+1])
            else:
                donch_high[i] = high_1d[i]
                donch_low[i] = low_1d[i]
        else:
            donch_high[i] = np.max(high_1d[i-19:i+1])
            donch_low[i] = np.min(low_1d[i-19:i+1])
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            vol_ma[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    volume_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need enough data for all indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_50_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = volume_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above Donchian high + 12h uptrend + volume confirmation
            if (price > donch_high_aligned[i] and 
                ema_12h_50_aligned[i] > ema_12h_50_aligned[i-1] and
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + 12h downtrend + volume confirmation
            elif (price < donch_low_aligned[i] and 
                  ema_12h_50_aligned[i] < ema_12h_50_aligned[i-1] and
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or 12h trend turns down
            if (price < donch_low_aligned[i] or 
                ema_12h_50_aligned[i] < ema_12h_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or 12h trend turns up
            if (price > donch_high_aligned[i] or 
                ema_12h_50_aligned[i] > ema_12h_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0