#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ChoppinessIndex_Touch_Pivot_R1S1_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d data for pivot and choppiness ---
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot and R1/S1 from previous daily bar
    prev_close_d = df_1d['close'].shift(1).values
    prev_high_d = df_1d['high'].shift(1).values
    prev_low_d = df_1d['low'].shift(1).values
    
    pivot_d = (prev_high_d + prev_low_d + prev_close_d) / 3
    range_d = prev_high_d - prev_low_d
    R1_d = pivot_d + range_d
    S1_d = pivot_d - range_d
    
    # Align daily R1/S1/pivot to 4h (wait for daily close)
    R1_d_aligned = align_htf_to_ltf(prices, df_1d, R1_d)
    S1_d_aligned = align_htf_to_ltf(prices, df_1d, S1_d)
    pivot_d_aligned = align_htf_to_ltf(prices, df_1d, pivot_d)
    
    # Choppiness Index (14) on daily
    atr_d = np.zeros(len(df_1d))
    tr_d = np.maximum(df_1d['high'] - df_1d['low'],
                      np.maximum(np.abs(df_1d['high'] - df_1d['close'].shift(1)),
                                 np.abs(df_1d['low'] - df_1d['close'].shift(1))))
    atr_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    highest_high_d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    chop_d = 100 * np.log10(atr_d * 14 / (highest_high_d - lowest_low_d)) / np.log10(14)
    chop_d_aligned = align_htf_to_ltf(prices, df_1d, chop_d, additional_delay_bars=0)  # chop uses current bar
    
    # Volume filter: current volume > 1.5 * 24-period average (6 days in 4h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_d_aligned[i]) or np.isnan(S1_d_aligned[i]) or
            np.isnan(pivot_d_aligned[i]) or np.isnan(chop_d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        R1_val = R1_d_aligned[i]
        S1_val = S1_d_aligned[i]
        pivot_val = pivot_d_aligned[i]
        chop_val = chop_d_aligned[i]
        vol_filter = volume_filter[i]
        
        # Only trade in ranging markets (Chop > 61.8)
        if chop_val <= 61.8:
            # Force flat in trending markets
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: touch or slightly penetrate S1 with volume in ranging market
            if close_val <= S1_val * 1.005 and vol_filter:  # Allow 0.5% penetration
                signals[i] = 0.25
                position = 1
            # Short: touch or slightly penetrate R1 with volume in ranging market
            elif close_val >= R1_val * 0.995 and vol_filter:  # Allow 0.5% penetration
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches or crosses pivot
            if close_val >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches or crosses pivot
            if close_val <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals