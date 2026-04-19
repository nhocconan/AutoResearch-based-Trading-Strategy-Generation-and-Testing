#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Camarilla_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate weekly ATR(14) for Camarilla width
    tr1 = np.maximum(high_1w[1:], close_1w[:-1]) - np.minimum(low_1w[1:], close_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly Camarilla levels using previous week's data
    prev_close = np.concatenate([[np.nan], close_1w[:-1]])
    prev_high = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low = np.concatenate([[np.nan], low_1w[:-1]])
    
    camarilla_H4 = prev_close + 1.1/2 * (prev_high - prev_low)
    camarilla_L4 = prev_close - 1.1/2 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_L4)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Need volume MA and Camarilla levels
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above weekly H4 with volume
            if price > camarilla_H4_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly L4 with volume
            elif price < camarilla_L4_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below weekly H4
            if price < camarilla_H4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above weekly L4
            if price > camarilla_L4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals