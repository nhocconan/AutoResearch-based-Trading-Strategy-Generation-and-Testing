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
    
    # Get 4h data for calculations (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour Donchian channels (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate rolling max/min with proper handling
    donchian_high = np.full(len(high_4h), np.nan)
    donchian_low = np.full(len(low_4h), np.nan)
    
    for i in range(19, len(high_4h)):  # 20-period window
        donchian_high[i] = np.max(high_4h[i-19:i+1])
        donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        alpha = 2 / (50 + 1)
        ema_50_4h[0] = close_4h[0]
        for i in range(1, len(close_4h)):
            ema_50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema_50_4h[i-1]
    
    # Align 4h indicators to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 4
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Session filter: 8-20 UTC (pre-compute hour array)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    size = 0.20
    
    # Warmup period
    start_idx = max(19, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and above 4h EMA50
            if price > donchian_high_aligned[i] and vol_filter and price > ema_50_4h_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume and below 4h EMA50
            elif price < donchian_low_aligned[i] and vol_filter and price < ema_50_4h_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian low or below 4h EMA50
            if price < donchian_low_aligned[i] or price < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian high or above 4h EMA50
            if price > donchian_high_aligned[i] or price > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Donchian_20_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0