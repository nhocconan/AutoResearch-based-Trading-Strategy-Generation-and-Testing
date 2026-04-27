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
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate rolling max/min with proper handling
    donchian_high_1d = np.full(len(high_1d), np.nan)
    donchian_low_1d = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):  # 20-period window
        donchian_high_1d[i] = np.max(high_1d[i-19:i+1])
        donchian_low_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        alpha = 2 / (34 + 1)
        ema_34_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    
    # Align 1d indicators to 6h timeframe
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1-week Donchian channels (10-period high/low) for context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min with proper handling
    donchian_high_1w = np.full(len(high_1w), np.nan)
    donchian_low_1w = np.full(len(low_1w), np.nan)
    
    for i in range(9, len(high_1w)):  # 10-period window
        donchian_high_1w[i] = np.max(high_1w[i-9:i+1])
        donchian_low_1w[i] = np.min(low_1w[i-9:i+1])
    
    # Align 1w indicators to 6h timeframe
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Calculate 4-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 4
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(19, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        # Weekly trend filter: price above/below weekly Donchian mid
        weekly_mid = (donchian_high_1w_aligned[i] + donchian_low_1w_aligned[i]) / 2
        weekly_uptrend = price > weekly_mid
        weekly_downtrend = price < weekly_mid
        
        if position == 0:
            # Long: Price breaks above daily Donchian high with volume and in weekly uptrend
            if price > donchian_high_1d_aligned[i] and vol_filter and weekly_uptrend:
                signals[i] = size
                position = 1
            # Short: Price breaks below daily Donchian low with volume and in weekly downtrend
            elif price < donchian_low_1d_aligned[i] and vol_filter and weekly_downtrend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below daily Donchian low or weekly trend turns down
            if price < donchian_low_1d_aligned[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above daily Donchian high or weekly trend turns up
            if price > donchian_high_1d_aligned[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_20_1dEMA34_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0