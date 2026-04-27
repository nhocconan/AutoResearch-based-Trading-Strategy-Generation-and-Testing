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
    
    # Get daily data for Bollinger Bands and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands (20, 2)
    sma_20 = np.full(len(close_1d), np.nan)
    std_20 = np.full(len(close_1d), np.nan)
    for i in range(20-1, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-19:i+1])
        std_20[i] = np.std(close_1d[i-19:i+1])
    
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Align Bollinger Bands to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate 50-period EMA on daily close for trend filter
    ema_50_1d = np.full(len(close_1d), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_1d)):
        if i < 49:
            ema_50_1d[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[i]
        else:
            if np.isnan(ema_50_1d[i-1]):
                ema_50_1d[i] = np.mean(close_1d[i-49:i+1])
            else:
                ema_50_1d[i] = close_1d[i] * alpha + ema_50_1d[i-1] * (1 - alpha)
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume ratio: current volume / 20-period average volume
    vol_ma_20 = np.full(len(volume), np.nan)
    for i in range(20-1, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    vol_ratio = np.full(n, np.nan)
    valid_vol = vol_ma_20 > 0
    vol_ratio[valid_vol] = volume[valid_vol] / vol_ma_20[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper BB + volume spike + price above daily EMA50
            if (price > upper_bb_aligned[i] and 
                vol_ratio[i] > 1.5 and 
                price > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB + volume spike + price below daily EMA50
            elif (price < lower_bb_aligned[i] and 
                  vol_ratio[i] > 1.5 and 
                  price < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below lower BB or trend weakens
            if (price < lower_bb_aligned[i] or 
                price < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above upper BB or trend weakens
            if (price > upper_bb_aligned[i] or 
                price > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerBreakout_TrendVolume_v1"
timeframe = "6h"
leverage = 1.0