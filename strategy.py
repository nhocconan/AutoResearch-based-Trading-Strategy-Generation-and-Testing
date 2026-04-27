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
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Donchian channels (20-period)
    donchian_high_1d = np.full(len(df_1d), np.nan)
    donchian_low_1d = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i >= 19:
            donchian_high_1d[i] = np.max(high_1d[i-19:i+1])
            donchian_low_1d[i] = np.min(low_1d[i-19:i+1])
        elif i >= 0:
            donchian_high_1d[i] = np.max(high_1d[:i+1])
            donchian_low_1d[i] = np.min(low_1d[:i+1])
    
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Calculate daily 20-period SMA for trend filter
    sma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:
            sma_20_1d[i] = np.mean(close_1d[i-19:i+1])
        elif i >= 0:
            sma_20_1d[i] = np.mean(close_1d[:i+1])
    
    sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
    
    # Calculate daily volume average (20-period)
    vol_avg_20_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:
            vol_avg_20_1d[i] = np.mean(volume_1d[i-19:i+1])
        elif i >= 0:
            vol_avg_20_1d[i] = np.mean(volume_1d[:i+1])
    
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Get weekly data for trend filter: EMA(34) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_34 = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (34 + 1)
    for i in range(len(close_1w)):
        if i < 33:
            ema_1w_34[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_1w_34[i-1]):
                ema_1w_34[i] = np.mean(close_1w[i-33:i+1])
            else:
                ema_1w_34[i] = close_1w[i] * alpha_w + ema_1w_34[i-1] * (1 - alpha_w)
    
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_1d_aligned[i]) or 
            np.isnan(donchian_low_1d_aligned[i]) or
            np.isnan(sma_20_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(ema_1w_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x daily average
        volume_confirm = vol > 1.5 * vol_avg_20_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume confirmation + weekly uptrend
            if (price > donchian_high_1d_aligned[i] and 
                volume_confirm and 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume confirmation + weekly downtrend
            elif (price < donchian_low_1d_aligned[i] and 
                  volume_confirm and 
                  ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below Donchian low or weekly trend turns down
            if (price < donchian_low_1d_aligned[i] or 
                ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian high or weekly trend turns up
            if (price > donchian_high_1d_aligned[i] or 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_DonchianBreakout_Volume_WeeklyEMA34_v1"
timeframe = "1d"
leverage = 1.0