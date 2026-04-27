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
    
    # Get weekly data for primary trend filter: EMA(21)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(21) weekly
    ema_1w_21 = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (21 + 1)
    for i in range(len(close_1w)):
        if i < 20:
            ema_1w_21[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_1w_21[i-1]):
                ema_1w_21[i] = np.mean(close_1w[i-20:i+1])
            else:
                ema_1w_21[i] = close_1w[i] * alpha_w + ema_1w_21[i-1] * (1 - alpha_w)
    
    ema_1w_21_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_21)
    
    # Get daily data for entry timing and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian Channel (20) on daily
    donchian_high = np.full(len(df_1d), np.nan)
    donchian_low = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        donchian_high[i] = np.max(high_1d[i-19:i+1])
        donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate volume ratio: current volume / 20-day average volume
    vol_ma20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma20[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    volume_ratio = np.full(n, np.nan)
    valid_vol = (~np.isnan(vol_ma20_aligned)) & (vol_ma20_aligned > 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma20_aligned[valid_vol]
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need at least 20 days for Donchian + 21 weeks for EMA
    start_idx = max(20, 21)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_1w_21_aligned[i]) or
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above Donchian high + weekly uptrend + volume
            if (price > donchian_high_aligned[i] and 
                ema_1w_21_aligned[i] > ema_1w_21_aligned[i-1] and
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + weekly downtrend + volume
            elif (price < donchian_low_aligned[i] and 
                  ema_1w_21_aligned[i] < ema_1w_21_aligned[i-1] and
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below Donchian low or weekly trend turns down
            if (price < donchian_low_aligned[i] or 
                ema_1w_21_aligned[i] < ema_1w_21_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian high or weekly trend turns up
            if (price > donchian_high_aligned[i] or 
                ema_1w_21_aligned[i] > ema_1w_21_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_DonchianBreakout_WeeklyEMA21_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0