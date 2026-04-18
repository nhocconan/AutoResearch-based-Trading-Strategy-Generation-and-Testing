#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_DailyPivot_R1S1_Breakout_VolumeATRFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for pivot levels and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot, R1, S1 from previous daily bar
    prev_close_d = df_1d['close'].shift(1).values
    prev_high_d = df_1d['high'].shift(1).values
    prev_low_d = df_1d['low'].shift(1).values
    
    pivot_d = (prev_high_d + prev_low_d + prev_close_d) / 3
    range_d = prev_high_d - prev_low_d
    R1_d = pivot_d + range_d
    S1_d = pivot_d - range_d
    
    # Align daily pivot levels to 4h (wait for daily close)
    R1_d_aligned = align_htf_to_ltf(prices, df_1d, R1_d)
    S1_d_aligned = align_htf_to_ltf(prices, df_1d, S1_d)
    pivot_d_aligned = align_htf_to_ltf(prices, df_1d, pivot_d)
    
    # Daily ATR for stop loss
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_d_aligned = align_htf_to_ltf(prices, df_1d, atr_d)
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(R1_d_aligned[i]) or np.isnan(S1_d_aligned[i]) or 
            np.isnan(pivot_d_aligned[i]) or np.isnan(atr_d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = R1_d_aligned[i]
        s1_val = S1_d_aligned[i]
        pivot_val = pivot_d_aligned[i]
        atr_val = atr_d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above R1 with volume
            if close_val > r1_val and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: break below S1 with volume
            elif close_val < s1_val and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: stop loss or price returns to pivot
            if close_val <= entry_price - 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
            elif close_val < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or price returns to pivot
            if close_val >= entry_price + 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
            elif close_val > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals