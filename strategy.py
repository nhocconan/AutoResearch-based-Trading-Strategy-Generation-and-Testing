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
    
    # Load 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA20 for trend
    ema_20_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        ema_20_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_20_1d[i] = (close_1d[i] * 2 + ema_20_1d[i-1] * 18) / 20
    
    # Calculate daily ATR(14) for volatility
    atr_14_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 14:
        tr = np.zeros(len(close_1d))
        for i in range(1, len(close_1d)):
            tr[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
        atr_14_1d[13] = np.mean(tr[1:14])
        for i in range(14, len(close_1d)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate Donchian channel (20-day)
    upper_dc = np.full_like(close_1d, np.nan)
    lower_dc = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        for i in range(19, len(close_1d)):
            upper_dc[i] = np.max(high_1d[i-19:i+1])
            lower_dc[i] = np.min(low_1d[i-19:i+1])
    
    # Align 1d indicators to 12h timeframe
    ema_20_1d_12h = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    atr_14_1d_12h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    upper_dc_12h = align_htf_to_ltf(prices, df_1d, upper_dc)
    lower_dc_12h = align_htf_to_ltf(prices, df_1d, lower_dc)
    
    # Calculate 12h volume MA for confirmation
    vol_ma_10 = np.full_like(volume, np.nan)
    if len(volume) >= 10:
        for i in range(9, len(volume)):
            vol_ma_10[i] = np.mean(volume[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1d_12h[i]) or 
            np.isnan(atr_14_1d_12h[i]) or
            np.isnan(upper_dc_12h[i]) or 
            np.isnan(lower_dc_12h[i]) or
            np.isnan(vol_ma_10[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 10-period average
        if vol_ma_10[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_10[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume and above EMA20
            if (close[i] > upper_dc_12h[i] and
                close[i] > ema_20_1d_12h[i] and
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower Donchian with volume and below EMA20
            elif (close[i] < lower_dc_12h[i] and
                  close[i] < ema_20_1d_12h[i] and
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls below lower Donchian or below EMA20
            if (close[i] < lower_dc_12h[i] or 
                close[i] < ema_20_1d_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises above upper Donchian or above EMA20
            if (close[i] > upper_dc_12h[i] or 
                close[i] > ema_20_1d_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian20_EMA20_Volume"
timeframe = "12h"
leverage = 1.0