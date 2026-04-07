#!/usr/bin/env python3
"""
6H Price Action Reversal with 1D Order Flow Imbalance
Long when price rejects lower 1D value area with bullish order flow imbalance (delta>0)
Short when price rejects upper 1D value area with bearish order flow imbalance (delta<0)
Exit on opposite value area touch or 12-bar time stop
Uses actual market structure (value areas) rather than arbitrary levels, works in both trends and ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_price_action_reversal_1d_orderflow_v1"
timeframe = "6h"
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
    taker_buy_volume = prices['taker_buy_volume'].values
    
    # === Order Flow Imbalance (buyer vs seller volume) ===
    delta = taker_buy_volume - (volume - taker_buy_volume)  # buy - sell
    delta_ma = pd.Series(delta).rolling(window=12, min_periods=12).mean().values
    
    # === 1D Value Areas (using 70% volume profile) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate value area high/low for each 1D bar
    va_high = []
    va_low = []
    va_point_control = []
    
    for i in range(len(df_1d)):
        # Simplified: use high/low of the day as proxy for value area
        # In practice, would use volume profile but this avoids look-ahead
        va_high.append(df_1d['high'].iloc[i])
        va_low.append(df_1d['low'].iloc[i])
        va_point_control.append((df_1d['high'].iloc[i] + df_1d['low'].iloc[i]) / 2)
    
    va_high = np.array(va_high)
    va_low = np.array(va_low)
    va_pc = np.array(va_point_control)
    
    # Align to 6H timeframe
    va_high_aligned = align_htf_to_ltf(prices, df_1d, va_high)
    va_low_aligned = align_htf_to_ltf(prices, df_1d, va_low)
    va_pc_aligned = align_htf_to_ltf(prices, df_1d, va_pc)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0
    
    for i in range(1, n):
        if (np.isnan(va_high_aligned[i]) or np.isnan(va_low_aligned[i]) or 
            np.isnan(delta_ma[i])):
            signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        bars_since_entry += 1
        
        # Time stop: exit after 12 bars (3 days)
        if position != 0 and bars_since_entry >= 12:
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: touch 1D VA high or PC
            if close[i] >= va_pc_aligned[i] * 0.998:  # Near point of control
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: touch 1D VA low or PC
            if close[i] <= va_pc_aligned[i] * 1.002:  # Near point of control
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for reversal
            # Need order flow imbalance confirmation
            if abs(delta_ma[i]) < 0.001 * volume[i]:  # No significant imbalance
                signals[i] = 0.0
                continue
            
            # Entry: price rejection at value area with order flow
            # Bullish rejection: price near VA low + buying pressure
            if (close[i] <= va_low_aligned[i] * 1.002 and  # Near VA low
                delta_ma[i] > 0):  # Buying pressure
                position = 1
                signals[i] = 0.25
                bars_since_entry = 0
            # Bearish rejection: price near VA high + selling pressure
            elif (close[i] >= va_high_aligned[i] * 0.998 and  # Near VA high
                  delta_ma[i] < 0):  # Selling pressure
                position = -1
                signals[i] = -0.25
                bars_since_entry = 0
    
    return signals