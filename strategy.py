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
    
    # Daily data (primary)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly data (HTF for trend)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === Daily Donchian channel (20-period) ===
    donchian_high = np.zeros_like(close_1d)
    donchian_low = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 20:
            donchian_high[i] = np.max(high_1d[:i+1])
            donchian_low[i] = np.min(low_1d[:i+1])
        else:
            donchian_high[i] = np.max(high_1d[i-19:i+1])
            donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    # === Weekly pivot points for trend filter ===
    weekly_pivot = np.zeros_like(close_1w)
    for i in range(1, len(close_1w)):
        pp = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
        weekly_pivot[i] = pp
    
    weekly_trend = np.where(close_1w > weekly_pivot, 1, -1)
    
    # === Daily volume ratio for confirmation ===
    vol_ma_10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_ratio_1d = volume_1d / vol_ma_10_1d
    
    # Align all HTF data to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Donchian and volume
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        wt = weekly_trend_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR weekly trend turns bearish
            if price < dl or wt == -1:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR weekly trend turns bullish
            if price > dh or wt == 1:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Donchian breakout above high with volume and weekly uptrend
            if price > dh and wt == 1 and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Donchian breakout below low with volume and weekly downtrend
            elif price < dl and wt == -1 and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0