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
    
    # Load weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend
    ema_200_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[:200])
        for i in range(200, len(df_1w)):
            ema_200_1w[i] = (close_1w[i] * 2 + ema_200_1w[i-1] * 198) / 200
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Load daily data for ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Load 12h data for Donchian channel
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian(20) on 12h
    upper_12h = np.full(len(df_12h), np.nan)
    lower_12h = np.full(len(df_12h), np.nan)
    
    for i in range(19, len(df_12h)):
        upper_12h[i] = np.max(high_12h[i-19:i+1])
        lower_12h[i] = np.min(low_12h[i-19:i+1])
    
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Volume spike detection (12h)
    vol_ma_10 = np.full_like(volume, np.nan)
    if len(volume) >= 10:
        for i in range(9, len(volume)):
            vol_ma_10[i] = np.mean(volume[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(atr_12h[i]) or
            np.isnan(upper_12h_aligned[i]) or
            np.isnan(lower_12h_aligned[i]) or
            np.isnan(vol_ma_10[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: require minimum volatility
        if atr_12h[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 10-period average
        if vol_ma_10[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_10[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 1.8
        
        if position == 0:
            # Long: Price breaks above Donchian upper with volume + weekly uptrend
            if (close[i] > upper_12h_aligned[i] and 
                volume_ratio > vol_threshold and
                close[i] > ema_200_1w_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian lower with volume + weekly downtrend
            elif (close[i] < lower_12h_aligned[i] and 
                  volume_ratio > vol_threshold and
                  close[i] < ema_200_1w_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below Donchian lower
            if close[i] < lower_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above Donchian upper
            if close[i] > upper_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_EMA200_12h_Donchian20_Volume_Trend"
timeframe = "12h"
leverage = 1.0