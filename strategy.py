#!/usr/bin/env python3
# 1D_WEEKLY_RANGE_BREAKOUT_VOLUME_CONFIRMATION
# Hypothesis: Breakout above/below weekly range (high-low) with volume confirmation
# works in both bull and bear markets. Weekly range acts as support/resistance.
# In uptrend, break above weekly high signals continuation; in downtrend, break below
# weekly low signals continuation. Volume filter ensures breakout is genuine.
# Uses 1d timeframe with 1h weekly range to avoid look-ahead.
# Target: 15-25 trades/year.

name = "1D_WEEKLY_RANGE_BREAKOUT_VOLUME_CONFIRMATION"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1h data for weekly range calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate weekly high and low (168 hours = 7 days)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    vol_1h = df_1h['volume'].values
    
    window = 168  # 7 days * 24 hours
    # Rolling max/min for weekly range
    weekly_high = pd.Series(high_1h).rolling(window=window, min_periods=window).max().values
    weekly_low = pd.Series(low_1h).rolling(window=window, min_periods=window).min().values
    # Average volume over same window for confirmation
    avg_volume = pd.Series(vol_1h).rolling(window=window, min_periods=window).mean().values
    
    # Align weekly range to 1d timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1h, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1h, weekly_low)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1h, avg_volume)
    
    # Current day's volume
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 168  # Need full first week
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(avg_volume_aligned[i]) or np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average weekly volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_aligned[i]
        
        if position == 0:
            # LONG: Break above weekly high with volume confirmation
            if (prices['close'][i] > weekly_high_aligned[i] and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly low with volume confirmation
            elif (prices['close'][i] < weekly_low_aligned[i] and volume_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly range or breaks below weekly low
            if (prices['close'][i] < weekly_high_aligned[i] or 
                prices['close'][i] < weekly_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly range or breaks above weekly high
            if (prices['close'][i] > weekly_low_aligned[i] or 
                prices['close'][i] > weekly_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals