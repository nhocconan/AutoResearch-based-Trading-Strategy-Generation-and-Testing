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
    
    # Load 4h data for multi-timeframe trend context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate 4h EMA(50) for trend
    ema_50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = (close_4h[i] * 0.0377) + (ema_50_4h[i-1] * 0.9623)
    
    # Align 4h EMA to 1h
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for volatility and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for volatility filter
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
    
    atr_1h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d volume average (20-period)
    vol_avg_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        for i in range(19, len(df_1d)):
            vol_avg_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_avg_1h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr_1h[i]) or
            np.isnan(vol_avg_1h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 1% of price)
        if atr_1h[i] < 0.01 * close[i]:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 50% of 1d average)
        if vol_avg_1h[i] <= 0 or volume[i] < 0.5 * vol_avg_1h[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 4h EMA(50)
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: Price above 4h EMA(50) with volume confirmation
            if trend_up:
                position = 1
                signals[i] = position_size
            # Short: Price below 4h EMA(50) with volume confirmation
            elif trend_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls below 4h EMA(50)
            if not trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises above 4h EMA(50)
            if trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_EMA50_1d_Volume_Filter"
timeframe = "1h"
leverage = 1.0