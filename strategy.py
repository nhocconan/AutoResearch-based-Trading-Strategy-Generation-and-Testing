#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Pivot_R1S1_Breakout_VolumeATRFilter_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for pivot and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Previous daily OHLC
    prev_close_d = df_1d['close'].shift(1).values
    prev_high_d = df_1d['high'].shift(1).values
    prev_low_d = df_1d['low'].shift(1).values
    
    # Pivot levels: R1, S1
    pivot_d = (prev_high_d + prev_low_d + prev_close_d) / 3
    range_d = prev_high_d - prev_low_d
    R1_d = pivot_d + range_d
    S1_d = pivot_d - range_d
    
    # ATR(14) for volatility filter
    tr1 = prev_high_d - prev_low_d
    tr2 = np.abs(prev_high_d - prev_close_d)
    tr3 = np.abs(prev_low_d - prev_close_d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align to 12h
    R1_d_aligned = align_htf_to_ltf(prices, df_1d, R1_d)
    S1_d_aligned = align_htf_to_ltf(prices, df_1d, S1_d)
    pivot_d_aligned = align_htf_to_ltf(prices, df_1d, pivot_d)
    atr_d_aligned = align_htf_to_ltf(prices, df_1d, atr_d)
    
    # Volume filter: current volume > 1.5 * 2-period average (2 * 12h = 1 day)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    volume_filter = volume > (1.5 * vol_ma_2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_d_aligned[i]) or np.isnan(S1_d_aligned[i]) or
            np.isnan(pivot_d_aligned[i]) or np.isnan(atr_d_aligned[i]) or
            np.isnan(vol_ma_2[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        R1_val = R1_d_aligned[i]
        S1_val = S1_d_aligned[i]
        pivot_val = pivot_d_aligned[i]
        atr_val = atr_d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above R1 with volume and ATR filter (avoid low-vol breakouts)
            if close_val > R1_val and vol_filter and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and ATR filter
            elif close_val < S1_val and vol_filter and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below pivot
            if close_val < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above pivot
            if close_val > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals