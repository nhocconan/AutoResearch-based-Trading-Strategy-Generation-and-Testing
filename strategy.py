#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels from previous week
    prev_close_1w = np.roll(close_1w, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w = np.roll(high_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w = np.roll(low_1w, 1)
    prev_low_1w[0] = np.nan
    
    # Camarilla R1 and S1 (weekly)
    r1_1w = prev_close_1w + (prev_high_1w - prev_low_1w) * 1.1 / 12.0
    s1_1w = prev_close_1w - (prev_high_1w - prev_low_1w) * 1.1 / 12.0
    
    # Align to daily timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Daily volume confirmation: current volume > 1.8x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.8x average
        volume_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume spike
            if price > r1_1w_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume spike
            elif price < s1_1w_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below weekly S1 (reversal signal)
            if price < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above weekly R1 (reversal signal)
            if price > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals