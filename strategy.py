#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Reversal_v1
Hypothesis: Weekly pivot reversals with volume confirmation on 6h timeframe.
- Calculates weekly pivot points (P), resistance (R1,R2), support (S1,S2) from prior week's OHLC.
- Enters long when price breaks above R1 with volume spike and price above weekly P (bullish bias).
- Enters short when price breaks below S1 with volume spike and price below weekly P (bearish bias).
- Exits when price crosses the weekly pivot point P (mean reversion).
- Works in bull/bear markets by using weekly pivot as dynamic support/resistance and volume to filter false breakouts.
"""

name = "6h_Weekly_Pivot_Reversal_v1"
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
    
    # === WEEKLY Data for Pivot Points ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's OHLC for pivot calculation
    ph_1w = high_1w  # previous week high
    pl_1w = low_1w   # previous week low
    pc_1w = close_1w # previous week close
    
    # Weekly pivot points: P = (H+L+C)/3
    pivot = (ph_1w + pl_1w + pc_1w) / 3.0
    # Resistance and support levels
    r1 = 2 * pivot - pl_1w
    s1 = 2 * pivot - ph_1w
    r2 = pivot + (ph_1w - pl_1w)
    s2 = pivot - (ph_1w - pl_1w)
    
    # Align weekly pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # === Volume Filter: 2.0x 20-period EMA on 6h ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and bullish bias (price > pivot)
            if (close[i] > r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > pivot_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and bearish bias (price < pivot)
            elif (close[i] < s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < pivot_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below weekly pivot (mean reversion)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above weekly pivot (mean reversion)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals