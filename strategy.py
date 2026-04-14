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
    
    # Load daily data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate daily pivot points (classic)
    if len(high_1d) < 1:
        return np.zeros(n)
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Calculate weekly pivot points (classic)
    if len(high_1w) < 1:
        return np.zeros(n)
    
    pivot_w = (high_1w + low_1w + close_1w) / 3.0
    r1_w = 2 * pivot_w - low_1w
    s1_w = 2 * pivot_w - high_1w
    
    # Align daily pivots
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Align weekly pivots
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # Calculate 50-period EMA for trend filter (daily)
    if len(close_1d) < 50:
        return np.zeros(n)
    
    ema50_1d = np.full_like(close_1d, np.nan)
    ema50_1d[49] = np.mean(close_1d[:50])
    for i in range(50, len(close_1d)):
        ema50_1d[i] = close_1d[i] * 0.0392 + ema50_1d[i-1] * 0.9608  # alpha = 2/(50+1)
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 20-period EMA for trend filter (weekly)
    if len(close_1w) < 20:
        return np.zeros(n)
    
    ema20_1w = np.full_like(close_1w, np.nan)
    ema20_1w[19] = np.mean(close_1w[:20])
    for i in range(20, len(close_1w)):
        ema20_1w[i] = close_1w[i] * 0.0952 + ema20_1w[i-1] * 0.9048  # alpha = 2/(20+1)
    
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate volume ratio (current 4h volume vs 20-period average)
    vol_ma_20 = np.full_like(volume, np.nan)
    for j in range(19, len(volume)):
        vol_ma_20[j] = np.mean(volume[j-19:j+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_w_aligned[i]) or 
            np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price above both daily and weekly S1 + price above both EMAs + volume surge
            if (close[i] > s1_aligned[i] and
                close[i] > s1_w_aligned[i] and
                close[i] > ema50_1d_aligned[i] and
                close[i] > ema20_1w_aligned[i] and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price below both daily and weekly R1 + price below both EMAs + volume surge
            elif (close[i] < r1_aligned[i] and
                  close[i] < r1_w_aligned[i] and
                  close[i] < ema50_1d_aligned[i] and
                  close[i] < ema20_1w_aligned[i] and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price below either daily or weekly S1 OR price below either EMA
            if (close[i] < s1_aligned[i] or 
                close[i] < s1_w_aligned[i] or
                close[i] < ema50_1d_aligned[i] or
                close[i] < ema20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price above either daily or weekly R1 OR price above either EMA
            if (close[i] > r1_aligned[i] or 
                close[i] > r1_w_aligned[i] or
                close[i] > ema50_1d_aligned[i] or
                close[i] > ema20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_1w_Pivot_S1S1_EMA50EMA20_Volume"
timeframe = "4h"
leverage = 1.0