#!/usr/bin/env python3
# 6h_weekly_pivot_volume_breakout_v1
# Hypothesis: 6h strategy using 1w Camarilla pivot levels with volume confirmation.
# Long: price breaks above weekly H3 with volume > 1.8x average volume
# Short: price breaks below weekly L3 with volume > 1.8x average volume
# Exit: price reverses to weekly H4/L4 levels
# Uses weekly pivots for structural support/resistance; volume confirms institutional participation.
# Designed for lower frequency (12-30 trades/year) to minimize fee drag on 6h timeframe.
# Weekly timeframe captures major market structure; 6h provides timely execution.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Use previous week's OHLC for Camarilla calculation
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + (range_val * 1.1 / 4)
    L3 = pivot - (range_val * 1.1 / 4)
    H4 = pivot + (range_val * 1.1 / 2)
    L4 = pivot - (range_val * 1.1 / 2)
    
    # Align HTF levels to LTF
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1w, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1w, L4)
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        if np.isnan(vol_r):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        h3 = H3_aligned[i]
        l3 = L3_aligned[i]
        h4 = H4_aligned[i]
        l4 = L4_aligned[i]
        
        if np.isnan(h3) or np.isnan(l3):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if price < h4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price > l4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price > h3 and vol_r > 1.8:
                position = 1
                signals[i] = 0.25
            elif price < l3 and vol_r > 1.8:
                position = -1
                signals[i] = -0.25
    
    return signals