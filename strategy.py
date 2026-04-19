#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_R1S1_Breakout_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h pivot levels from previous 4h bar
    prev_close_4h = np.roll(close_4h, 1)
    prev_close_4h[0] = np.nan
    prev_high_4h = np.roll(high_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h = np.roll(low_4h, 1)
    prev_low_4h[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_4h = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_4h = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 12.0
    
    # Align to 1h timeframe
    pivot_4h_1h = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_1h = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_1h = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(pivot_4h_1h[i]) or np.isnan(r1_4h_1h[i]) or np.isnan(s1_4h_1h[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0x average
        volume_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Price breaks above 4h R1 with volume spike
            if price > r1_4h_1h[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h S1 with volume spike
            elif price < s1_4h_1h[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: Price returns below 4h S1 (reversal signal)
            if price < s1_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Price returns above 4h R1 (reversal signal)
            if price > r1_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals