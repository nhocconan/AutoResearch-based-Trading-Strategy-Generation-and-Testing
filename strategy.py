#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for pivot and ATR
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Daily ATR (14)
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Daily Pivot Points (Standard)
    pivot_daily = (high_daily + low_daily + close_daily) / 3.0
    r1_daily = 2 * pivot_daily - low_daily
    s1_daily = 2 * pivot_daily - high_daily
    r2_daily = pivot_daily + (high_daily - low_daily)
    s2_daily = pivot_daily - (high_daily - low_daily)
    
    pivot_daily_aligned = align_htf_to_ltf(prices, df_daily, pivot_daily)
    r1_daily_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    r2_daily_aligned = align_htf_to_ltf(prices, df_daily, r2_daily)
    s2_daily_aligned = align_htf_to_ltf(prices, df_daily, s2_daily)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(pivot_daily_aligned[i]) or np.isnan(r1_daily_aligned[i]) or 
            np.isnan(s1_daily_aligned[i]) or np.isnan(atr_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        pivot = pivot_daily_aligned[i]
        r1 = r1_daily_aligned[i]
        s1 = s1_daily_aligned[i]
        r2 = r2_daily_aligned[i]
        s2 = s2_daily_aligned[i]
        atr_daily = atr_daily_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma_recent = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma_recent
        
        if position == 0:
            # Long: break above R1 with volume, target R2
            if price > r1 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume, target S2
            elif price < s1 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches R2 or falls back below pivot
            if price >= r2 or price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S2 or rises back above pivot
            if price <= s2 or price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Pivot_R1S1_Breakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0