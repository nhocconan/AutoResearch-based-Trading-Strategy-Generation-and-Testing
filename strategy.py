#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(df_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 49) / 51
    
    # Load daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Donchian(20) channels
    high_max_20 = np.full(len(df_1d), np.nan)
    low_min_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        high_max_20[i] = np.max(high_1d[i-19:i+1])
        low_min_20[i] = np.min(low_1d[i-19:i+1])
    
    # Load daily ATR for volatility filter
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
    
    # Align weekly EMA and daily indicators to 6h timeframe
    ema_50_1w_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    high_max_20_6h = align_htf_to_ltf(prices, df_1d, high_max_20)
    low_min_20_6h = align_htf_to_ltf(prices, df_1d, low_min_20)
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike detection (20-period average on 6h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1w_6h[i]) or 
            np.isnan(high_max_20_6h[i]) or
            np.isnan(low_min_20_6h[i]) or
            np.isnan(atr_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_6h[i] < 0.003 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.5
        
        # Determine trend direction from weekly EMA
        uptrend = close[i] > ema_50_1w_6h[i]
        downtrend = close[i] < ema_50_1w_6h[i]
        
        if position == 0:
            # Long: Price breaks above weekly EMA AND Donchian high with volume confirmation
            if (uptrend and 
                close[i] > high_max_20_6h[i] and 
                volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below weekly EMA AND Donchian low with volume confirmation
            elif (downtrend and 
                  close[i] < low_min_20_6h[i] and 
                  volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below weekly EMA
            if close[i] < ema_50_1w_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above weekly EMA
            if close[i] > ema_50_1w_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_EMA50_1d_Donchian20_Volume"
timeframe = "6h"
leverage = 1.0