#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_R4_S4_Breakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    
    # Previous weekly OHLC
    prev_close_w = df_1w['close'].shift(1).values
    prev_high_w = df_1w['high'].shift(1).values
    prev_low_w = df_1w['low'].shift(1).values
    
    # Pivot levels: R4, S4
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    range_w = prev_high_w - prev_low_w
    R4_w = pivot_w + 3 * range_w
    S4_w = pivot_w - 3 * range_w
    
    # Align to daily
    R4_w_aligned = align_htf_to_ltf(prices, df_1w, R4_w)
    S4_w_aligned = align_htf_to_ltf(prices, df_1w, S4_w)
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    
    # Volume filter: current volume > 1.5 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R4_w_aligned[i]) or np.isnan(S4_w_aligned[i]) or
            np.isnan(pivot_w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        R4_val = R4_w_aligned[i]
        S4_val = S4_w_aligned[i]
        pivot_val = pivot_w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above R4 with volume
            if close_val > R4_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S4 with volume
            elif close_val < S4_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below weekly pivot
            if close_val < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above weekly pivot
            if close_val > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals