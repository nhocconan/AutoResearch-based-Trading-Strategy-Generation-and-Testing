#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-period)
    high_20 = np.full_like(high_1w, np.nan)
    low_20 = np.full_like(low_1w, np.nan)
    for i in range(20, len(high_1w)):
        high_20[i] = np.max(high_1w[i-20:i])
        low_20[i] = np.min(low_1w[i-20:i])
    
    # Weekly EMA50 for trend filter
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Weekly volume average for confirmation
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    # Align weekly indicators to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price breaks below weekly lower band or volume drops
            if price < low_20_aligned[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above weekly upper band or volume drops
            if price > high_20_aligned[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly upper band with volume and uptrend
            if price > high_20_aligned[i] and vol_ratio > 2.0 and price > ema_50_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly lower band with volume and downtrend
            elif price < low_20_aligned[i] and vol_ratio > 2.0 and price < ema_50_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals