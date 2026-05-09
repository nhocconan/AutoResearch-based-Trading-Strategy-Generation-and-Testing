#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_VolumeSpike_CamarillaPivot_Reversal"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily high, low, close for Camarilla pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla levels: H3, L3, H4, L4
    H3 = daily_close + 1.1 * (daily_high - daily_low) / 6
    L3 = daily_close - 1.1 * (daily_high - daily_low) / 6
    H4 = daily_close + 1.1 * (daily_high - daily_low) / 2
    L4 = daily_close - 1.1 * (daily_high - daily_low) / 2
    
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume spike: current volume > 2x 20-period SMA
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or \
           np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or \
           np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Fade at H3/L3 with volume spike
            if price >= H3_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            elif price <= L3_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Breakout continuation at H4/L4 with volume spike
            elif price > H4_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif price < L4_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below H3 or volume spike fades
            if price < H3_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above L3 or volume spike fades
            if price > L3_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals