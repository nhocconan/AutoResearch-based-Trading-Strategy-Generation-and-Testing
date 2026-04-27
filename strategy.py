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
    
    # Get weekly data for trend filter (weekly EMA200)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 200:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA200
    ema_period = 200
    ema_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= ema_period:
        ema_weekly[ema_period - 1] = np.mean(close_weekly[:ema_period])
        for i in range(ema_period, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * (2 / (ema_period + 1)) + 
                            ema_weekly[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align weekly EMA200 to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Get daily data for Donchian breakout and volume filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Calculate 20-day Donchian channels
    high_20 = np.full(len(high_daily), np.nan)
    low_20 = np.full(len(low_daily), np.nan)
    
    for i in range(20-1, len(high_daily)):
        high_20[i] = np.max(high_daily[i-20+1:i+1])
        low_20[i] = np.min(low_daily[i-20+1:i+1])
    
    # Align Donchian channels to daily timeframe (already aligned)
    high_20_aligned = high_20
    low_20_aligned = low_20
    
    # Calculate 20-day average volume
    vol_ma_20 = np.full(len(volume_daily), np.nan)
    for i in range(20, len(volume_daily)):
        vol_ma_20[i] = np.mean(volume_daily[i-20:i])
    
    # Align volume MA to daily timeframe
    vol_ma_20_aligned = vol_ma_20
    
    # Align all daily data to minute timeframe
    high_20_min = align_htf_to_ltf(prices, df_daily, high_20_aligned)
    low_20_min = align_htf_to_ltf(prices, df_daily, low_20_aligned)
    vol_ma_20_min = align_htf_to_ltf(prices, df_daily, vol_ma_20_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly EMA200 and daily Donchian
    start_idx = max(200, 20) + 1  # extra buffer for calculations
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(high_20_min[i]) or 
            np.isnan(low_20_min[i]) or 
            np.isnan(vol_ma_20_min[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20_min[i] if vol_ma_20_min[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above 20-day high + volume spike + price > weekly EMA200
            if (price > high_20_min[i] and 
                vol_ratio > 2.0 and 
                price > ema_weekly_aligned[i]):
                signals[i] = size
                position = 1
            # Short: Price breaks below 20-day low + volume spike + price < weekly EMA200
            elif (price < low_20_min[i] and 
                  vol_ratio > 2.0 and 
                  price < ema_weekly_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below 20-day low OR price < weekly EMA200
            if (price < low_20_min[i] or 
                price < ema_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price breaks above 20-day high OR price > weekly EMA200
            if (price > high_20_min[i] or 
                price > ema_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyEMA200_DonchianBreakout_VolumeSpike"
timeframe = "1d"
leverage = 1.0