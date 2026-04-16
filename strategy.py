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
    
    # === Daily OHLC for pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Pivot Points (Standard) ===
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # === True Range and ATR (14) on daily ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Align to 12h timeframe ===
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === Volume spike detection (10-period) ===
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    warmup = 50
    
    for i in range(warmup, n):
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(atr_12h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        atr = atr_12h[i]
        vol_spike = volume_spike[i]
        
        # === Exit Conditions ===
        if position == 1:  # Long
            if price < s1_12h[i] or price > r2_12h[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short
            if price > r1_12h[i] or price < s2_12h[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # === Entry Conditions ===
        if position == 0:
            # Long: Price crosses above R1 with volume spike
            if price > r1_12h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price crosses below S1 with volume spike
            elif price < s1_12h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_VolumeSpike"
timeframe = "12h"
leverage = 1.0