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
    
    # Calculate weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Calculate weekly pivot points
    pivot_w = (high_1w + low_1w + close_1w) / 3.0
    r1_w = 2 * pivot_w - low_1w
    s1_w = 2 * pivot_w - high_1w
    r2_w = pivot_w + (high_1w - low_1w)
    s2_w = pivot_w - (high_1w - low_1w)
    
    # Align daily pivot levels to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Align weekly pivot levels to 12h timeframe
    pivot_w_12h = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_12h = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_12h = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_w_12h = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_12h = align_htf_to_ltf(prices, df_1w, s2_w)
    
    # Calculate 14-period weekly ATR for volatility filter
    tr_w = np.zeros(len(df_1w))
    tr_w[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr_w[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_w[13] = np.mean(tr_w[:14])
        for i in range(14, len(df_1w)):
            atr_w[i] = (atr_w[i-1] * 13 + tr_w[i]) / 14
    
    atr_w_12h = align_htf_to_ltf(prices, df_1w, atr_w)
    
    # Volume spike detection (20-period average on 12h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_12h[i]) or 
            np.isnan(r1_12h[i]) or
            np.isnan(s1_12h[i]) or
            np.isnan(r2_12h[i]) or
            np.isnan(s2_12h[i]) or
            np.isnan(pivot_w_12h[i]) or
            np.isnan(r1_w_12h[i]) or
            np.isnan(s1_w_12h[i]) or
            np.isnan(r2_w_12h[i]) or
            np.isnan(s2_w_12h[i]) or
            np.isnan(atr_w_12h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_w_12h[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.5
        
        if position == 0:
            # Long: Price breaks above R2 (daily) AND R2 (weekly) with volume confirmation
            if (close[i] > r2_12h[i] and close[i] > r2_w_12h[i] and volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S2 (daily) AND S2 (weekly) with volume confirmation
            elif (close[i] < s2_12h[i] and close[i] < s2_w_12h[i] and volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below S1 (daily) OR S1 (weekly)
            if close[i] < s1_12h[i] or close[i] < s1_w_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above R1 (daily) OR R1 (weekly)
            if close[i] > r1_12h[i] or close[i] > r1_w_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_1w_Pivot_R2S2_Breakout_Volume"
timeframe = "12h"
leverage = 1.0