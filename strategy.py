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
    
    # Load weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    
    # Align pivot levels to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_12h = align_htf_to_ltf(prices, df_1w, r1)
    s1_12h = align_htf_to_ltf(prices, df_1w, s1)
    r2_12h = align_htf_to_ltf(prices, df_1w, r2)
    s2_12h = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate 14-period weekly ATR for volatility filter
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    atr_12h = align_htf_to_ltf(prices, df_1w, atr_1w)
    
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
            np.isnan(atr_12h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.8% of price)
        if atr_12h[i] < 0.008 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above R2 with volume confirmation
            if (close[i] > r2_12h[i] and volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S2 with volume confirmation
            elif (close[i] < s2_12h[i] and volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below S1
            if close[i] < s1_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above R1
            if close[i] > r1_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Pivot_R2S2_Breakout_Volume"
timeframe = "12h"
leverage = 1.0