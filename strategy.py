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
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-day Donchian high and low
    donch_high = np.full(len(df_1d), np.nan)
    donch_low = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        donch_high[i] = np.max(high_1d[i-19:i+1])
        donch_low[i] = np.min(low_1d[i-19:i+1])
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
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
    
    # Calculate weekly volume average for volume confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_1w_10 = np.full(len(df_1w), np.nan)
    for i in range(len(vol_1w)):
        if i < 9:
            vol_ma_1w_10[i] = np.mean(vol_1w[:i+1]) if i > 0 else vol_1w[i]
        else:
            vol_ma_1w_10[i] = np.mean(vol_1w[i-9:i+1])
    
    vol_ma_1w_10_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w_10)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(19, 34, 9)
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(ema_1w_34_aligned[i]) or
            np.isnan(vol_ma_1w_10_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_1w_10_aligned[i]
        
        # Volume confirmation: current volume > 1.5x weekly average
        vol_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend + volume
            if (price > donch_high_aligned[i] and 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1] and
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + weekly downtrend + volume
            elif (price < donch_low_aligned[i] and 
                  ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1] and
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or weekly trend turns down
            if (price < donch_low_aligned[i] or 
                ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or weekly trend turns up
            if (price > donch_high_aligned[i] or 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DonchianBreakout_WeeklyEMA34_VolumeConfirmed_v1"
timeframe = "12h"
leverage = 1.0