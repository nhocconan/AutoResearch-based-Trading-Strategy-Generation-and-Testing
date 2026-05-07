#!/usr/bin/env python3
name = "6h_PivotBreakout_VolumeTrend_1d"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily pivot points (classic)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = 2 * pivot - df_1d['low']
    s1 = 2 * pivot - df_1d['high']
    r2 = pivot + (df_1d['high'] - df_1d['low'])
    s2 = pivot - (df_1d['high'] - df_1d['low'])
    r3 = r1 + (df_1d['high'] - df_1d['low'])
    s3 = s1 - (df_1d['high'] - df_1d['low'])
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Volume spike filter (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # 6-period price momentum (rate of change)
    roc = np.zeros_like(close)
    roc[6:] = (close[6:] - close[:-6]) / close[:-6]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 6)
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(roc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike and positive momentum
            if close[i] > r1_aligned[i] and vol_spike[i] and roc[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume spike and negative momentum
            elif close[i] < s1_aligned[i] and vol_spike[i] and roc[i] < 0:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to pivot or momentum reverses
            if position == 1:
                if close[i] < pivot_aligned[i] or roc[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_aligned[i] or roc[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals