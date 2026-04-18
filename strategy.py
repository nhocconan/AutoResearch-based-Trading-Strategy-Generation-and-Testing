# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_BB_Width_Squeeze_Breakout_With_Volume
6h strategy using Bollinger Band width squeeze + breakout with volume confirmation.
- Bollinger Band width (BBW) calculated on 1d close (20-period, 2 std)
- BBW squeeze: current BBW < 20-period lowest BBW (volatility contraction)
- Breakout: price breaks above/below 20-period Bollinger Bands
- Volume confirmation: volume > 1.5x 20-period average
- Long: squeeze + breakout up + volume confirmation
- Short: squeeze + breakout down + volume confirmation
- Exit: opposite signal or price crosses 20-period Bollinger middle band (SMA20)
- Designed for low frequency: ~10-20 trades/year per symbol (40-80 total over 4 years)
- Works in both bull and bear markets by capturing volatility breakouts after consolidation
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
    
    # Get 1d data for Bollinger Bands calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb  # Band width
    
    # Bollinger Band width squeeze: current width < 20-period lowest width
    bb_width_min = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    bb_squeeze = bb_width < bb_width_min
    
    # Align BB squeeze and Bollinger Bands to 6h timeframe
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need 20 for BB calculations + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_squeeze_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(sma_20_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > upper_bb_aligned[i]
        breakout_down = close[i] < lower_bb_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: BB squeeze + breakout up + volume confirmation
            if bb_squeeze_aligned[i] and breakout_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze + breakout down + volume confirmation
            elif bb_squeeze_aligned[i] and breakout_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: opposite signal or price crosses below SMA20 (mean reversion)
            if bb_squeeze_aligned[i] and breakout_down and volume_confirm or close[i] < sma_20_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: opposite signal or price crosses above SMA20 (mean reversion)
            if bb_squeeze_aligned[i] and breakout_up and volume_confirm or close[i] > sma_20_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BB_Width_Squeeze_Breakout_With_Volume"
timeframe = "6h"
leverage = 1.0