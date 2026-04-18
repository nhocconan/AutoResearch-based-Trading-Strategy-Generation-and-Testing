#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using weekly pivot points (from 1w data) for directional bias,
combined with 6h Donchian(15) breakout and volume confirmation. Weekly pivots provide
strong institutional support/resistance levels. In bull markets, we buy dips to weekly S1/S2;
in bear markets, we sell rallies to weekly R1/R2. Volume confirms institutional participation.
Designed for low trade frequency (15-25/year) to minimize fee drag in ranging/bear markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H-L), S2 = P - (H-L)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    
    # Align weekly pivots to 6h (weekly pivots are fixed for the week)
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate 6h Donchian(15) channels
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(15, n):
        highest_high[i] = np.max(high[i-15:i])
        lowest_low[i] = np.min(low[i-15:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(15, 20)  # need Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(vol_ma[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price near weekly support (S1 or S2) with bullish bias
            # Bullish bias: price above weekly pivot
            near_support = (abs(low[i] - s1_6h[i]) < 0.1 * (r1_6h[i] - s1_6h[i]) or
                           abs(low[i] - s2_6h[i]) < 0.1 * (r1_6h[i] - s1_6h[i]))
            bullish_bias = close[i] > pivot_6h[i]
            
            if near_support and bullish_bias and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price near weekly resistance (R1 or R2) with bearish bias
            # Bearish bias: price below weekly pivot
            elif (abs(high[i] - r1_6h[i]) < 0.1 * (r1_6h[i] - s1_6h[i]) or
                  abs(high[i] - r2_6h[i]) < 0.1 * (r1_6h[i] - s1_6h[i])):
                bearish_bias = close[i] < pivot_6h[i]
                if bearish_bias and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price reaches weekly resistance or breaks below support
            if (high[i] >= r1_6h[i] or low[i] < s2_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches weekly support or breaks above resistance
            if (low[i] <= s1_6h[i] or high[i] > r2_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_S1R1_Donchian15_Volume"
timeframe = "6h"
leverage = 1.0