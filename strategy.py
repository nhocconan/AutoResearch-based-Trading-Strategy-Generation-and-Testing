#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Pivot_S1R1_Breakout_VolumeATR_Tight"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot points and ATR
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(14)
    tr1 = np.maximum(high_12h[1:], close_12h[:-1]) - np.minimum(low_12h[1:], close_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # 12h pivot points: P = (H+L+C)/3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    s1_12h = 2 * pivot_12h - high_12h
    r1_12h = 2 * pivot_12h - low_12h
    
    # Align to 4h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    
    # Volume confirmation: current volume > 1.5x 30-period average (on 4h)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(r1_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_30[i]
        atr = atr_12h_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        s1 = s1_12h_aligned[i]
        r1 = r1_12h_aligned[i]
        
        if position == 0:
            # Long: Break above R1 with volume
            if price > r1 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume
            elif price < s1 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below S1 or ATR stop (2.0x ATR)
            if price < s1 or price < (high[i] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R1 or ATR stop (2.0x ATR)
            if price > r1 or price > (low[i] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals