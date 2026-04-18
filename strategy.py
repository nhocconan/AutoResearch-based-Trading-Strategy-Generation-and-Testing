#!/usr/bin/env python3
"""
6h_WeeklyPivot_Direction_1dVolumeFilter
Hypothesis: Use weekly pivot levels for long-term trend direction, with 1d volume confirmation for entry timing. 
Go long when price is above weekly pivot (bullish bias) and breaks above 1d high with volume > 1.5x average. 
Go short when price is below weekly pivot (bearish bias) and breaks below 1d low with volume > 1.5x average. 
Weekly pivot provides structural bias that works in both bull and bear markets, while volume confirmation ensures 
entries occur with conviction. Target: 20-40 trades/year by requiring both directional bias and volume confirmation.
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
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly pivot points: P = (H + L + C)/3
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    # Align weekly pivot to 6h timeframe
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    
    # Get daily data for high/low and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Align daily high/low to 6h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_weekly_aligned[i]) or 
            np.isnan(high_1d_aligned[i]) or 
            np.isnan(low_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above weekly pivot AND breaks above 1d high with volume
            if close[i] > pivot_weekly_aligned[i] and high[i] > high_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot AND breaks below 1d low with volume
            elif close[i] < pivot_weekly_aligned[i] and low[i] < low_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly pivot OR breaks below 1d low
            if close[i] < pivot_weekly_aligned[i] or low[i] < low_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly pivot OR breaks above 1d high
            if close[i] > pivot_weekly_aligned[i] or high[i] > high_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Direction_1dVolumeFilter"
timeframe = "6h"
leverage = 1.0