#!/usr/bin/env python3
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
    
    # Get 12h data for calculations (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour Donchian channels (20-period high/low)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate rolling max/min with proper handling
    donchian_high = np.full(len(high_12h), np.nan)
    donchian_low = np.full(len(low_12h), np.nan)
    
    for i in range(19, len(high_12h)):  # 20-period window
        donchian_high[i] = np.max(high_12h[i-19:i+1])
        donchian_low[i] = np.min(low_12h[i-19:i+1])
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        alpha = 2 / (50 + 1)
        ema_50_12h[0] = close_12h[0]
        for i in range(1, len(close_12h)):
            ema_50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_50_12h[i-1]
    
    # Align 12h indicators to 12h timeframe (same timeframe, no alignment needed)
    donchian_high_aligned = donchian_high
    donchian_low_aligned = donchian_low
    ema_50_12h_aligned = ema_50_12h
    
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
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and above 12h EMA50
            if price > donchian_high_aligned[i] and vol_filter and price > ema_50_12h_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume and below 12h EMA50
            elif price < donchian_low_aligned[i] and vol_filter and price < ema_50_12h_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian low or below 12h EMA50
            if price < donchian_low_aligned[i] or price < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian high or above 12h EMA50
            if price > donchian_high_aligned[i] or price > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_20_12hEMA50_Volume"
timeframe = "12h"
leverage = 1.0