#!/usr/bin/env python3
"""
6h_12h_Camarilla_Pivot_Breakout_Volume_Confirmation
Hypothesis: Camarilla pivot levels from 12h act as strong support/resistance. Breakouts above R3 or below S3 with volume confirmation indicate institutional interest and trend continuation. Works in bull markets by capturing breakouts and in bear markets by fading false breaks at R3/S3 and confirming real breaks with volume. Targets 15-30 trades/year.
"""

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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    R4 = np.zeros_like(close_12h)
    R3 = np.zeros_like(close_12h)
    S3 = np.zeros_like(close_12h)
    S4 = np.zeros_like(close_12h)
    
    for i in range(1, len(close_12h)):
        # Use previous 12h bar's high, low, close to calculate today's levels
        H = high_12h[i-1]
        L = low_12h[i-1]
        C = close_12h[i-1]
        diff = H - L
        R3[i] = C + (diff * 1.1 / 4)
        S3[i] = C - (diff * 1.1 / 4)
        R4[i] = C + (diff * 1.1 / 2)
        S4[i] = C - (diff * 1.1 / 2)
    
    # Align 12h Camarilla levels to 6h timeframe (with 1-bar delay for completed bar)
    R3_6h = align_htf_to_ltf(prices, df_12h, R3)
    S3_6h = align_htf_to_ltf(prices, df_12h, S3)
    R4_6h = align_htf_to_ltf(prices, df_12h, R4)
    S4_6h = align_htf_to_ltf(prices, df_12h, S4)
    
    # Volume confirmation: 20-period EMA on 6h
    volume_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Volume EMA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R3_6h[i]) or 
            np.isnan(S3_6h[i]) or 
            np.isnan(R4_6h[i]) or 
            np.isnan(S4_6h[i]) or 
            np.isnan(volume_ema20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period EMA
        volume_filter = volume[i] > (1.5 * volume_ema20[i])
        
        if position == 0:
            # Long breakout: price breaks above R3 with volume, target R4
            if close[i] > R3_6h[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S3 with volume, target S4
            elif close[i] < S3_6h[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches R4 or reverses below R3
            if close[i] >= R4_6h[i] or close[i] < R3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S4 or reverses above S3
            if close[i] <= S4_6h[i] or close[i] > S3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_Camarilla_Pivot_Breakout_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0