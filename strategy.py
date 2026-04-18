#!/usr/bin/env python3
"""
1d_1W_Camarilla_Pivot_Breakout_Volume
Hypothesis: Use weekly Camarilla pivot levels on 1d chart with volume confirmation.
Go long when price breaks above weekly H3 resistance with volume > 1.5x 20-day average,
short when price breaks below weekly L3 support with volume > 1.5x 20-day average.
Exit on opposite side break (L3 for long, H3 for short).
Weekly pivots provide structure that works in both bull and bear markets by capturing
institutional levels. Volume filter reduces false breakouts. Target: 15-25 trades/year.
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
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (H3, L3)
    # H3 = close + 1.1*(high - low)/2
    # L3 = close - 1.1*(high - low)/2
    H3_1w = close_1w + 1.1 * (high_1w - low_1w) / 2
    L3_1w = close_1w - 1.1 * (high_1w - low_1w) / 2
    
    # Align weekly Camarilla levels to 1d timeframe
    H3_1w_aligned = align_htf_to_ltf(prices, df_1w, H3_1w)
    L3_1w_aligned = align_htf_to_ltf(prices, df_1w, L3_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1)  # Need at least one week of data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H3_1w_aligned[i]) or np.isnan(L3_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly H3 with volume confirmation
            if close[i] > H3_1w_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly L3 with volume confirmation
            elif close[i] < L3_1w_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly L3
            if close[i] < L3_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly H3
            if close[i] > H3_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Camarilla_Pivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0