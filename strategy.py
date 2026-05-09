#!/usr/bin/env python3
# 1d_Donchian20_Breakout_1wTrend_VolumeSpike
# Hypothesis: Daily Donchian(20) breakouts filtered by weekly EMA50 trend and volume spike.
# Works in bull/bear: Weekly trend filter avoids counter-trend trades, volume spike confirms institutional interest.
# Donchian provides clear breakout levels, weekly EMA50 gives robust trend filter.
# Uses volume ratio (current/20-bar average) for confirmation.

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous 20-day high/low for Donchian channels
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Shift by 1 to use previous day's channels (avoid look-ahead)
    donchian_high = np.concatenate([[np.nan], donchian_high[:-1]])
    donchian_low = np.concatenate([[np.nan], donchian_low[:-1]])
    
    # Align Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (ema_50_1w[i-1] * 49 + close_1w[i]) / 50
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND uptrend (price > weekly EMA50) AND volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND downtrend (price < weekly EMA50) AND volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR trend reversal (price < weekly EMA50)
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR trend reversal (price > weekly EMA50)
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals