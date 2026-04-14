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
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR for volatility filter (14-period)
    tr_1w = np.zeros(len(df_1w))
    tr_1w[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr_1w[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr_1w[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    atr_12h = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily
    ema_50_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1d[0] = close_1d[0]
        for i in range(1, len(df_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly Donchian channels (20-period)
    high_20_1w = np.full(len(df_1w), np.nan)
    low_20_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 20:
        for i in range(19, len(df_1w)):
            high_20_1w[i] = np.max(high_1w[i-19:i+1])
            low_20_1w[i] = np.min(low_1w[i-19:i+1])
    
    high_20_12h = align_htf_to_ltf(prices, df_1w, high_20_1w)
    low_20_12h = align_htf_to_ltf(prices, df_1w, low_20_1w)
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_12h[i]) or 
            np.isnan(low_20_12h[i]) or
            np.isnan(ema_50_12h[i]) or
            np.isnan(atr_12h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_12h[i] < 0.003 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 3.0
        
        if position == 0:
            # Long: Price breaks above weekly high with volume confirmation and above weekly EMA50
            if (close[i] > high_20_12h[i] and 
                volume_ratio > vol_threshold and
                close[i] > ema_50_12h[i]):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below weekly low with volume confirmation and below weekly EMA50
            elif (close[i] < low_20_12h[i] and 
                  volume_ratio > vol_threshold and
                  close[i] < ema_50_12h[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below weekly low
            if close[i] < low_20_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above weekly high
            if close[i] > high_20_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Donchian_EMA50_Volume"
timeframe = "12h"
leverage = 1.0