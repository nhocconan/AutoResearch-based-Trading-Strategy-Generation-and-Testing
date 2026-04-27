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
    
    # Get 1d data for price channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate 20-day Donchian channels
    upper_20 = np.full(len(high_1d), np.nan)
    lower_20 = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):
        upper_20[i] = np.max(high_1d[i-19:i+1])
        lower_20[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 20-day average volume for volume filter
    vol_avg_20 = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):
        vol_avg_20[i] = np.mean(vol_1d[i-19:i+1])
    
    # Calculate 10-day RSI for momentum filter
    delta = np.diff(close_1d)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(len(gain), np.nan)
    avg_loss = np.full(len(loss), np.nan)
    
    for i in range(10, len(gain)):
        if i == 10:
            avg_gain[i] = np.mean(gain[1:11])
            avg_loss[i] = np.mean(loss[1:11])
        else:
            avg_gain[i] = (avg_gain[i-1] * 9 + gain[i]) / 10
            avg_loss[i] = (avg_loss[i-1] * 9 + loss[i]) / 10
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_10 = 100 - (100 / (1 + rs))
    
    # Align indicators to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    rsi_10_aligned = align_htf_to_ltf(prices, df_1d, rsi_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian channels, volume average, and RSI
    start_idx = max(19, 10) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(rsi_10_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_avg_20_aligned[i] if vol_avg_20_aligned[i] > 0 else 0
        rsi = rsi_10_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 20-day high + volume confirmation + RSI > 50
            if (price > upper_20_aligned[i] and 
                vol_ratio > 1.5 and 
                rsi > 50):
                signals[i] = size
                position = 1
            # Short: Price breaks below 20-day low + volume confirmation + RSI < 50
            elif (price < lower_20_aligned[i] and 
                  vol_ratio > 1.5 and 
                  rsi < 50):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below 20-day low OR RSI < 30
            if (price < lower_20_aligned[i] or rsi < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price breaks above 20-day high OR RSI > 70
            if (price > upper_20_aligned[i] or rsi > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_Volume_RSI"
timeframe = "12h"
leverage = 1.0