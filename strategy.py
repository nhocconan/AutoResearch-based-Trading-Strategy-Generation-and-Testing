#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_S1_S4_Breakout_Volume
Hypothesis: Trade Camarilla pivot breakouts on 12h with volume confirmation. Go long when price breaks above S1 with volume > 1.5x 24-period average, short when breaks below S4 with volume confirmation. Exit on opposite pivot level touch (S4 for longs, S1 for shorts). Uses 1d high/low/close to calculate Camarilla levels. Designed for low frequency (12-30 trades/year) to avoid fee drag, works in bull/bear by following institutional pivot levels that act as support/resistance in all markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # S1 = close - 1.083 * (high - low)
    # S4 = close - 1.500 * (high - low)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    camarilla_s4 = np.full_like(close_1d, np.nan)
    
    valid = (~np.isnan(high_1d)) & (~np.isnan(low_1d)) & (~np.isnan(close_1d))
    if np.any(valid):
        hl = high_1d - low_1d
        camarilla_s1[valid] = close_1d[valid] - 1.083 * hl[valid]
        camarilla_s4[valid] = close_1d[valid] - 1.500 * hl[valid]
    
    # Align Camarilla levels to 12h timeframe
    s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    s4_12h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(s1_12h[i]) or np.isnan(s4_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above S1 with volume
            if close[i] > s1_12h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume
            elif close[i] < s4_12h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches or breaks below S4
            if close[i] < s4_12h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches or breaks above S1
            if close[i] > s1_12h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_S1_S4_Breakout_Volume"
timeframe = "12h"
leverage = 1.0