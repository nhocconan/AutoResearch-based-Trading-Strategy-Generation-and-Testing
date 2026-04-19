#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Pivot_R1S1_Breakout_Volume_Spike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w pivot levels from previous 1w bar
    prev_close_1w = np.roll(close_1w, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w = np.roll(high_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w = np.roll(low_1w, 1)
    prev_low_1w[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_1w = prev_close_1w + (prev_high_1w - prev_low_1w) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_1w = prev_close_1w - (prev_high_1w - prev_low_1w) * 1.1 / 12.0
    
    # Align to 12h timeframe
    pivot_1w_12h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_12h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_12h = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1w_12h[i]) or np.isnan(r1_1w_12h[i]) or np.isnan(s1_1w_12h[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0x average
        volume_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Price breaks above 1w R1 with volume spike
            if price > r1_1w_12h[i] and volume_spike:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below 1w S1 with volume spike
            elif price < s1_1w_12h[i] and volume_spike:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit: Price returns below 1w S1 (reversal signal)
            if price < s1_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: Price returns above 1w R1 (reversal signal)
            if price > r1_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals