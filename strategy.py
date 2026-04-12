#!/usr/bin/env python3
"""
12h_1w_camarilla_volume_regime
Uses weekly Camarilla levels on 1w and daily volume confirmation on 12h.
Long when price approaches L3 support with rising volume, short when approaches H3 resistance with rising volume.
Exits at L4/H4 levels or when volume dries up.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
Works in both trending and ranging markets by combining institutional levels with volume confirmation.
"""

name = "12h_1w_camarilla_volume_regime"
timeframe = "12h"
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
    
    # Get weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 4:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each weekly bar
    L4 = np.full(len(close_1w), np.nan)
    L3 = np.full(len(close_1w), np.nan)
    L2 = np.full(len(close_1w), np.nan)
    L1 = np.full(len(close_1w), np.nan)
    H1 = np.full(len(close_1w), np.nan)
    H2 = np.full(len(close_1w), np.nan)
    H3 = np.full(len(close_1w), np.nan)
    H4 = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        if i == 0 or np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i]):
            continue
        range_val = high_1w[i] - low_1w[i]
        if range_val <= 0:
            continue
        close_val = close_1w[i]
        H4[i] = close_val + 1.5 * range_val
        H3[i] = close_val + 1.25 * range_val
        H2[i] = close_val + 1.166 * range_val
        H1[i] = close_val + 1.0833 * range_val
        L1[i] = close_val - 1.0833 * range_val
        L2[i] = close_val - 1.166 * range_val
        L3[i] = close_val - 1.25 * range_val
        L4[i] = close_val - 1.5 * range_val
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all levels and volume confirmation to 12h
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1w, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1w, L4)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_1d > (vol_ma_1d * 1.2))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: near L3 support with volume confirmation
        if (low[i] <= L3_aligned[i] * 1.005 and  # within 0.5% of L3
            vol_confirm_aligned[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: near H3 resistance with volume confirmation
        elif (high[i] >= H3_aligned[i] * 0.995 and  # within 0.5% of H3
              vol_confirm_aligned[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (high[i] >= H4_aligned[i] * 0.995 or  # near H4
                                not vol_confirm_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (low[i] <= L4_aligned[i] * 1.005 or  # near L4
                                 not vol_confirm_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals