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
    
    # Load daily data for pivot and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate daily pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align pivot levels to 1d timeframe (same as primary)
    pivot_1d = align_htf_to_ltf(prices, df_1d, pivot)
    r1_1d = align_htf_to_ltf(prices, df_1d, r1)
    s1_1d = align_htf_to_ltf(prices, df_1d, s1)
    r2_1d = align_htf_to_ltf(prices, df_1d, r2)
    s2_1d = align_htf_to_ltf(prices, df_1d, s2)
    r3_1d = align_htf_to_ltf(prices, df_1d, r3)
    s3_1d = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate daily ATR for volatility filter (14-period)
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
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike detection (20-day average on daily)
    vol_ma_20d = np.full(len(vol_1d), np.nan)
    if len(vol_1d) >= 20:
        for i in range(19, len(vol_1d)):
            vol_ma_20d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_1d[i]) or 
            np.isnan(r2_1d[i]) or
            np.isnan(s2_1d[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_20d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_1d_aligned[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 1d volume vs 20-day average
        if vol_ma_20d_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = vol_1d[i] / vol_ma_20d_aligned[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above R2 with volume confirmation
            if (close[i] > r2_1d[i] and 
                volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S2 with volume confirmation
            elif (close[i] < s2_1d[i] and 
                  volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below pivot
            if close[i] < pivot_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above pivot
            if close[i] > pivot_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Pivot_R2S2_Breakout_Volume"
timeframe = "1d"
leverage = 1.0