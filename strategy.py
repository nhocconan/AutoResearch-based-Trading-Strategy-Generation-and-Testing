#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1dPivot_R1S1_Breakout_VolumeATRFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot, R1, S1 from previous daily bar
    prev_close_d = df_1d['close'].shift(1).values
    prev_high_d = df_1d['high'].shift(1).values
    prev_low_d = df_1d['low'].shift(1).values
    
    pivot_d = (prev_high_d + prev_low_d + prev_close_d) / 3
    range_d = prev_high_d - prev_low_d
    R1_d = pivot_d + range_d
    S1_d = pivot_d - range_d
    
    # Align daily pivot levels to 12h (wait for daily close)
    pivot_d_aligned = align_htf_to_ltf(prices, df_1d, pivot_d)
    R1_d_aligned = align_htf_to_ltf(prices, df_1d, R1_d)
    S1_d_aligned = align_htf_to_ltf(prices, df_1d, S1_d)
    
    # Volume filter: current volume > 1.8 * 4-period average (2 days)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_filter = volume > (1.8 * vol_ma_4)
    
    # ATR for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_d_aligned[i]) or np.isnan(R1_d_aligned[i]) or
            np.isnan(S1_d_aligned[i]) or np.isnan(vol_ma_4[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pivot_val = pivot_d_aligned[i]
        R1_val = R1_d_aligned[i]
        S1_val = S1_d_aligned[i]
        vol_filter = volume_filter[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above R1 with volume
            if close_val > R1_val and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: break below S1 with volume
            elif close_val < S1_val and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: price falls back below pivot OR stop loss hit
            if close_val < pivot_val or close_val <= entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above pivot OR stop loss hit
            if close_val > pivot_val or close_val >= entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals