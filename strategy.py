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
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily 10-period EMA
    ema_10_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 10:
        ema_10_1d[9] = np.mean(close_1d[:10])
        multiplier = 2 / (10 + 1)
        for i in range(10, len(df_1d)):
            ema_10_1d[i] = (close_1d[i] - ema_10_1d[i-1]) * multiplier + ema_10_1d[i-1]
    
    # Calculate daily 50-period EMA
    ema_50_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        multiplier = 2 / (50 + 1)
        for i in range(50, len(df_1d)):
            ema_50_1d[i] = (close_1d[i] - ema_50_1d[i-1]) * multiplier + ema_50_1d[i-1]
    
    # Calculate daily ATR (14-period) - Wilder's smoothing
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 6-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 6-hour volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    # Align indicators to 6h timeframe
    ema_10_6h = align_htf_to_ltf(prices, df_1d, ema_10_1d)
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_10_6h[i]) or
            np.isnan(ema_50_6h[i]) or
            np.isnan(atr_6h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_6h[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 80% of 20-period MA)
        if volume[i] < 0.8 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 10 EMA > 50 EMA AND price breaks above Donchian high
            if ema_10_6h[i] > ema_50_6h[i] and close[i] > donch_high[i]:
                position = 1
                signals[i] = position_size
            # Short: 10 EMA < 50 EMA AND price breaks below Donchian low
            elif ema_10_6h[i] < ema_50_6h[i] and close[i] < donch_low[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: 10 EMA crosses below 50 EMA OR price falls below Donchian low
            if ema_10_6h[i] < ema_50_6h[i] or close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: 10 EMA crosses above 50 EMA OR price rises above Donchian high
            if ema_10_6h[i] > ema_50_6h[i] or close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_EMA10_50_Donchian20_VolumeFilter"
timeframe = "6h"
leverage = 1.0