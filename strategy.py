#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
- Uses Donchian breakout (20-bar high/low) for momentum capture
- Weekly pivot (from 1w data) determines long/short bias: only long above weekly pivot, short below
- Volume confirmation (1.5x 20-period average) filters false breakouts
- Designed for 6h timeframe: targets 50-150 total trades over 4 years (12-37/year)
- Works in bull markets (breakouts with upward bias) and bear markets (breakdowns with downward bias)
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) == 0:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: P = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bias from weekly pivot
        bullish_bias = close[i] > pivot_1w_aligned[i]
        bearish_bias = close[i] < pivot_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakdown_down = close[i] < lowest_low[i]
        
        if position == 0:
            # Long: bullish bias + volume + upward breakout
            if bullish_bias and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: bearish bias + volume + downward breakdown
            elif bearish_bias and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish bias or volume-confirmed breakdown
            if bearish_bias or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish bias or volume-confirmed breakout
            if bullish_bias or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0