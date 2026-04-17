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
    
    # === 1h Close for trend filter ===
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    # EMA50 on 1h
    close_1h_series = pd.Series(close_1h)
    ema_50_1h = close_1h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    # === 4h Donchian Channel (20) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Calculate Donchian bands
    high_max_20 = np.full(len(high_4h), np.nan)
    low_min_20 = np.full(len(low_4h), np.nan)
    for i in range(len(high_4h)):
        if i >= 19:
            high_max_20[i] = np.max(high_4h[i-19:i+1])
            low_min_20[i] = np.min(low_4h[i-19:i+1])
    # Align to 4h timeframe
    upper_donchian = align_htf_to_ltf(prices, df_4h, high_max_20)
    lower_donchian = align_htf_to_ltf(prices, df_4h, low_min_20)
    
    # === 1d Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    # Volume MA20
    vol_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_1h_aligned[i]) or np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 1.5
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper Donchian + price > 1h EMA50 + volume confirmation
            if high[i] > upper_donchian[i] and close[i] > ema_50_1h_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian + price < 1h EMA50 + volume confirmation
            elif low[i] < lower_donchian[i] and close[i] < ema_50_1h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price returns to middle of Donchian channel
        elif position == 1:
            # Exit long: price crosses below midpoint of Donchian
            midpoint = (upper_donchian[i] + lower_donchian[i]) / 2
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above midpoint of Donchian
            midpoint = (upper_donchian[i] + lower_donchian[i]) / 2
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_1hEMA50_Volume"
timeframe = "4h"
leverage = 1.0