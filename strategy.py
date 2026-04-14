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
    
    # Load 1-day data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily moving averages for trend filter
    ma_20_1d = np.full(len(df_1d), np.nan)
    ma_50_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        ma_20_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(df_1d)):
            ma_20_1d[i] = np.mean(close_1d[i-19:i+1])
    if len(df_1d) >= 50:
        ma_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(df_1d)):
            ma_50_1d[i] = np.mean(close_1d[i-49:i+1])
    
    # Align daily MAs to 12h timeframe
    ma_20_12h = align_htf_to_ltf(prices, df_1d, ma_20_1d)
    ma_50_12h = align_htf_to_ltf(prices, df_1d, ma_50_1d)
    
    # Calculate 12h Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate ATR for volatility filter (14-period)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = np.full(n, np.nan)
    if n >= 14:
        atr[13] = np.mean(tr[:14])
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Volume spike detection (20-period average)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or
            np.isnan(ma_20_12h[i]) or
            np.isnan(ma_50_12h[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and trend filter
            if (close[i] > donch_high[i] and 
                volume_ratio > vol_threshold and
                ma_20_12h[i] > ma_50_12h[i]):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian low with volume and trend filter
            elif (close[i] < donch_low[i] and 
                  volume_ratio > vol_threshold and
                  ma_20_12h[i] < ma_50_12h[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below Donchian low or trend reverses
            if (close[i] < donch_low[i] or 
                ma_20_12h[i] < ma_50_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above Donchian high or trend reverses
            if (close[i] > donch_high[i] or 
                ma_20_12h[i] > ma_50_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_MA_Volume_Filter"
timeframe = "12h"
leverage = 1.0