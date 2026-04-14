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
    
    # Load weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, R2 = P + (H - L), R3 = H + 2*(P - L)
    # S1 = 2*P - H, S2 = P - (H - L), S3 = L - 2*(H - P)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3
    r1_w = 2 * pivot_w - low_w
    r2_w = pivot_w + (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s1_w = 2 * pivot_w - high_w
    s2_w = pivot_w - (high_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    
    # Align weekly pivots to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_1w, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_1w, s3_w)
    
    # Load daily data for volume confirmation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day volume average
    volume_d = df_1d['volume'].values
    vol_ma_20 = np.full_like(volume_d, np.nan)
    for i in range(19, len(volume_d)):
        vol_ma_20[i] = np.mean(volume_d[i-19:i+1])
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 20-day EMA for trend filter
    close_d = df_1d['close'].values
    ema_20 = np.full_like(close_d, np.nan)
    if len(close_d) >= 20:
        ema_20[19] = np.mean(close_d[:20])
        for i in range(20, len(close_d)):
            ema_20[i] = close_d[i] * 0.1 + ema_20[i-1] * 0.9
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # Start after enough data for alignment
        # Skip if any critical data is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r3_w_aligned[i]) or 
            np.isnan(s3_w_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(ema_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-day average
        if vol_ma_20_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for long entries: break above weekly R3 with volume surge in uptrend
            if (close[i] > r3_w_aligned[i] and 
                volume_ratio > 2.0 and
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = position_size
            # Look for short entries: break below weekly S3 with volume surge in downtrend
            elif (close[i] < s3_w_aligned[i] and 
                  volume_ratio > 2.0 and
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below weekly pivot or trend reverses
            if (close[i] < pivot_w_aligned[i] or
                close[i] < ema_20_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above weekly pivot or trend reverses
            if (close[i] > pivot_w_aligned[i] or
                close[i] > ema_20_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_Breakout_Volume_Trend_v1"
timeframe = "6h"
leverage = 1.0