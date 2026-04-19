#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R1S1_Breakout_Volume_ATR"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 6-day ATR for volatility filter
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, 0.0)  # First TR is 0
    atr_period = 6
    atr = np.zeros_like(close)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[1:atr_period+1])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Volume filter: current volume > 1.3x 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 6)
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or \
           np.isnan(s1_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        atr_val = atr[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.3 * vol_ma
        
        # ATR filter: avoid extremely low volatility
        atr_ma = np.nanmean(atr[max(0, i-24):i+1]) if i >= 24 else atr_val
        volatility_ok = atr_val > 0.5 * atr_ma
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and volatility
            if price > r1 and volume_ok and volatility_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and volatility
            elif price < s1 and volume_ok and volatility_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to pivot or breaks below S1 (stop)
            if price < pivot or price < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to pivot or breaks above R1 (stop)
            if price > pivot or price > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals