#!/usr/bin/env python3
"""
4h_1d_InsideBarBreakout_WithVolume
Hypothesis: Use 4h timeframe with 1d inside bar breakouts confirmed by volume.
An inside bar (high < previous high AND low > previous low) indicates consolidation.
Breakout from this range with volume > 1.5x average captures momentum in both bull and bear markets.
Targets 20-40 trades/year by requiring tight consolidation and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "4h_1d_InsideBarBreakout_WithVolume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d inside bar: high < prev high AND low > prev low
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    inside_bar = (df_1d['high'].values < prev_high) & (df_1d['low'].values > prev_low)
    
    # Inside bar high/low for breakout levels
    inside_high = df_1d['high'].values
    inside_low = df_1d['low'].values
    
    # Align to 4h timeframe
    inside_bar_4h = align_ltf_to_htf(prices, df_1d, inside_bar)
    inside_high_4h = align_ltf_to_htf(prices, df_1d, inside_high)
    inside_low_4h = align_ltf_to_htf(prices, df_1d, inside_low)
    
    # Volume average (20 period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(inside_bar_4h[i]) or np.isnan(inside_high_4h[i]) or 
            np.isnan(inside_low_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Breakout conditions
        long_breakout = inside_bar_4h[i] and (high[i] > inside_high_4h[i]) and volume_spike
        short_breakout = inside_bar_4h[i] and (low[i] < inside_low_4h[i]) and volume_spike
        
        # Exit: return to opposite side of inside bar range
        long_exit = inside_bar_4h[i] and (low[i] < inside_low_4h[i])
        short_exit = inside_bar_4h[i] and (high[i] > inside_high_4h[i])
        
        # Priority: entry > exit > hold
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals