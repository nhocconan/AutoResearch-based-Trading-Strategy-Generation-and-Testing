#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point (P) = (H + L + C)/3
    pivot = (high_1w + low_1w + close_1w) / 3
    # Resistance 1 (R1) = 2*P - L
    r1 = 2 * pivot - low_1w
    # Support 1 (S1) = 2*P - H
    s1 = 2 * pivot - high_1w
    # Resistance 2 (R2) = P + (H - L)
    r2 = pivot + (high_1w - low_1w)
    # Support 2 (S2) = P - (H - L)
    s2 = pivot - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.5x daily volume MA (adjusted for 6h)
        # 4 6h periods per day, so daily MA/4 = approximate 6h period MA
        volume_6h_approx_ma = volume_ma_20_1d_aligned[i] / 4
        volume_condition = volume[i] > (volume_6h_approx_ma * 1.5)
        
        # Entry conditions: Fade at weekly pivot extremes with volume confirmation
        # Long when price touches S1/S2 with volume and shows rejection
        # Short when price touches R1/R2 with volume and shows rejection
        near_support = (low[i] <= s1_aligned[i] * 1.005) or (low[i] <= s2_aligned[i] * 1.005)
        near_resistance = (high[i] >= r1_aligned[i] * 0.995) or (high[i] >= r2_aligned[i] * 0.995)
        
        # Rejection signals: close back inside pivot range
        rejecting_support = close[i] > s1_aligned[i] and close[i] < pivot_aligned[i]
        rejecting_resistance = close[i] < r1_aligned[i] and close[i] > pivot_aligned[i]
        
        if position == 0:
            # Long setup: price rejects support with volume
            if near_support and rejecting_support and volume_condition:
                position = 1
                signals[i] = position_size
            # Short setup: price rejects resistance with volume
            elif near_resistance and rejecting_resistance and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price reaches pivot or shows rejection at resistance
            if close[i] >= pivot_aligned[i] or near_resistance:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price reaches pivot or shows rejection at support
            if close[i] <= pivot_aligned[i] or near_support:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_Fade_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0