#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1-week pivot levels and volume confirmation.
- Use weekly R3/S3 levels for mean reversion (fade extreme deviations)
- Use weekly R4/S4 levels for breakout continuation (strong momentum)
- Only trade when price is outside weekly pivot range (avoid chop)
- Volume > 1.5x average confirms institutional participation
- Works in bull/bear: fade extremes in ranging markets, follow breakouts in trending
Target: 15-35 trades/year per symbol (60-140 over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using previous week)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)
    
    pivot_1w = np.full(len(close_1w), np.nan)
    r1_1w = np.full(len(close_1w), np.nan)
    s1_1w = np.full(len(close_1w), np.nan)
    r2_1w = np.full(len(close_1w), np.nan)
    s2_1w = np.full(len(close_1w), np.nan)
    r3_1w = np.full(len(close_1w), np.nan)
    s3_1w = np.full(len(close_1w), np.nan)
    r4_1w = np.full(len(close_1w), np.nan)
    s4_1w = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        H = high_1w[i-1]
        L = low_1w[i-1]
        C = close_1w[i-1]
        
        if np.isnan(H) or np.isnan(L) or np.isnan(C):
            continue
            
        P = (H + L + C) / 3.0
        pivot_1w[i] = P
        r1_1w[i] = 2 * P - L
        s1_1w[i] = 2 * P - H
        r2_1w[i] = P + (H - L)
        s2_1w[i] = P - (H - L)
        r3_1w[i] = H + 2 * (P - L)
        s3_1w[i] = L - 2 * (H - P)
        r4_1w[i] = r3_1w[i] + (H - L)
        s4_1w[i] = s3_1w[i] - (H - L)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Volume confirmation (20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivots and volume MA
    start_idx = 20  # Need at least 20 bars for volume MA and week data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s4_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Get weekly levels
        pivot = pivot_1w_aligned[i]
        r3 = r3_1w_aligned[i]
        s3 = s3_1w_aligned[i]
        r4 = r4_1w_aligned[i]
        s4 = s4_1w_aligned[i]
        
        if position == 0:
            # Long conditions:
            # 1. Fade at S3: price < S3 and volume spike (mean reversion)
            # 2. Breakout at R4: price > R4 and volume spike (continuation)
            if vol_filter:
                if price < s3:
                    signals[i] = size
                    position = 1
                elif price > r4:
                    signals[i] = size
                    position = 1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
                
        elif position == 1:
            # Exit conditions:
            # 1. Price returns to pivot (mean reversion target)
            # 2. Price reaches opposite R3 (take profit)
            # 3. Price fails at R4 and reverses (for breakout trades)
            if price >= pivot or price >= r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
                
        elif position == -1:
            # Short conditions:
            # 1. Fade at R3: price > R3 and volume spike (mean reversion)
            # 2. Breakdown at S4: price < S4 and volume spike (continuation)
            if vol_filter:
                if price > r3:
                    signals[i] = -size
                    position = -1
                elif price < s4:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
                
    return signals

name = "6h_WeeklyPivot_R3S3_R4S4_Volume"
timeframe = "6h"
leverage = 1.0