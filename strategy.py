#!/usr/bin/env python3
"""
12h Camarilla Pivot with 1d Volume Spike and Chop Filter
Hypothesis: Camarilla pivot levels act as strong support/resistance. 
Price touching S3/R3 with volume spike and low chop indicates mean reversion.
In chop regime, we fade extreme moves; in trend, we wait for breakouts.
Works in both bull/bear by adapting to regime.
Target: 12-37 trades/year (50-150 total) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Chop filter (14-period)
    def calculate_chop(high, low, close, window=14):
        atr = []
        tr = []
        for i in range(len(close)):
            if i == 0:
                tr.append(high[i] - low[i])
            else:
                tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
            if i < window:
                atr.append(np.nan)
            else:
                atr.append(np.mean(tr[i-window+1:i+1]))
        atr = np.array(atr)
        
        max_h = np.full(len(close), np.nan)
        min_l = np.full(len(close), np.nan)
        for i in range(len(close)):
            if i < window:
                continue
            max_h[i] = np.max(high[i-window+1:i+1])
            min_l[i] = np.min(low[i-window+1:i+1])
        
        chop = np.full(len(close), np.nan)
        for i in range(window-1, len(close)):
            if atr[i] == 0 or max_h[i] == min_l[i]:
                chop[i] = 50
            else:
                chop[i] = 100 * np.log10(sum(tr[i-window+1:i+1]) / (max_h[i] - min_l[i])) / np.log10(window)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_filter = chop < 61.8  # Trending when chop < 61.8
    
    # Volume Spike Detector (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # 1d Camarilla Pivot Levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    s3 = pivot - (range_1d * 1.1 / 6)
    s4 = pivot - (range_1d * 1.1 / 2)
    r3 = pivot + (range_1d * 1.1 / 6)
    r4 = pivot + (range_1d * 1.1 / 2)
    
    # Align to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S4 or chop turns low (trending)
            if close[i] <= s4_aligned[i] or chop_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R4 or chop turns low (trending)
            if close[i] >= r4_aligned[i] or chop_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade in choppy market (chop >= 61.8)
            if chop_filter[i]:  # Trending regime, wait
                signals[i] = 0.0
                continue
                
            # Long: price touches S3 with volume spike
            if (low[i] <= s3_aligned[i] and 
                close[i] > s3_aligned[i] and  # Confirmed bounce
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 with volume spike
            elif (high[i] >= r3_aligned[i] and 
                  close[i] < r3_aligned[i] and  # Confirmed rejection
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals