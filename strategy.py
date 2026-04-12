#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_plus_volume
Breakout above/below Camarilla H3/L3 levels from daily timeframe with volume confirmation.
Enter on break of H3 (long) or L3 (short) with volume > 1.5x 20-period average.
Exit when price returns to Pivot point or opposite H4/L4 level.
Uses discrete position sizing (0.25) to limit risk and reduce trade frequency.
Designed for 4h timeframe with 1d Camarilla levels to capture multi-day breakouts.
Works in both bull and bear markets by following institutional pivot levels.
"""

name = "4h_1d_camarilla_breakout_plus_volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    typical = (high + low + close) / 3
    range_val = high - low
    # Camarilla levels
    H4 = close + range_val * 1.1 / 2
    H3 = close + range_val * 1.1 / 4
    L3 = close - range_val * 1.1 / 4
    L4 = close - range_val * 1.1 / 2
    return H3, L3, H4, L4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    H3_1d = np.full_like(close_1d, np.nan)
    L3_1d = np.full_like(close_1d, np.nan)
    H4_1d = np.full_like(close_1d, np.nan)
    L4_1d = np.full_like(close_1d, np.nan)
    P_1d = np.full_like(close_1d, np.nan)  # Pivot point
    
    for i in range(len(df_1d)):
        H3, L3, H4, L4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        H3_1d[i] = H3
        L3_1d[i] = L3
        H4_1d[i] = H4
        L4_1d[i] = L4
        P_1d[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
    
    # Align Camarilla levels to 4h timeframe
    H3_4h = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3_1d)
    H4_4h = align_htf_to_ltf(prices, df_1d, H4_1d)
    L4_4h = align_htf_to_ltf(prices, df_1d, L4_1d)
    P_4h = align_htf_to_ltf(prices, df_1d, P_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or 
            np.isnan(H4_4h[i]) or np.isnan(L4_4h[i]) or np.isnan(P_4h[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above H3 with volume confirmation
        if (close[i] > H3_4h[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below L3 with volume confirmation
        elif (close[i] < L3_4h[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Long exit: price returns to Pivot or breaks below L4
        elif position == 1 and (close[i] <= P_4h[i] or close[i] < L4_4h[i]):
            position = 0
            signals[i] = 0.0
        # Short exit: price returns to Pivot or breaks above H4
        elif position == -1 and (close[i] >= P_4h[i] or close[i] > H4_4h[i]):
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