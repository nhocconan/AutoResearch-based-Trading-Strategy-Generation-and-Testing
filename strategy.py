#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_v1
# Hypothesis: Uses weekly Camarilla pivot levels on daily chart with volume confirmation.
# Long when price crosses above weekly L4 (support) with volume > 1.5x average; short when price crosses below weekly H4 (resistance) with volume > 1.5x average.
# Weekly timeframe provides stable support/resistance levels that work in both bull and bear markets by fading overextensions at key levels.
# Target: 15-25 trades/year (60-100 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_v1"
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
    
    # Load weekly data ONCE before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels for previous week
    # Formula: based on previous week's high, low, close
    ph = df_1w['high'].values  # previous week high
    pl = df_1w['low'].values   # previous week low
    pc = df_1w['close'].values # previous week close
    
    # Camarilla levels
    # H4 = close + 1.5 * (high - low) * 1.1/2
    # L4 = close - 1.5 * (high - low) * 1.1/2
    range_1w = ph - pl
    h4 = pc + 1.5 * range_1w * 1.1 / 2
    l4 = pc - 1.5 * range_1w * 1.1 / 2
    
    # Align weekly Camarilla levels to daily timeframe (wait for previous week's close)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly L4 (support break)
            if close[i] < l4_aligned[i] and close[i-1] >= l4_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above weekly H4 (resistance break)
            if close[i] > h4_aligned[i] and close[i-1] <= h4_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price crosses above weekly L4 with volume confirmation
            if close[i] > l4_aligned[i] and close[i-1] <= l4_aligned[i-1] and vol_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: price crosses below weekly H4 with volume confirmation
            elif close[i] < h4_aligned[i] and close[i-1] >= h4_aligned[i-1] and vol_ok:
                position = -1
                signals[i] = -0.25
    
    return signals