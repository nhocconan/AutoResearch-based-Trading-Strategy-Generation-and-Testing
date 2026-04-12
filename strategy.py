#!/usr/bin/env python3
# 6h_1d_alligator_trend_with_volume
# Hypothesis: Combines Williams Alligator (13,8,5 SMAs) with volume confirmation on 6h timeframe.
# The Alligator identifies trend direction when jaws, teeth, and lips are aligned.
# Volume confirms trend strength. Works in both bull/bear by only trading when trend is clear.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

name = "6h_1d_alligator_trend_with_volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Williams Alligator lines: 13, 8, 5 period SMAs
    # Jaw (13-period SMA)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    # Teeth (8-period SMA)
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    # Lips (5-period SMA)
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw (alligator eating up)
        bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish alignment: Lips < Teeth < Jaw (alligator eating down)
        bearish = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Long entry: bullish alignment with volume confirmation
        if bullish and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: bearish alignment with volume confirmation
        elif bearish and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite alignment or loss of volume confirmation
        elif position == 1 and (not bullish or not vol_confirm[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish or not vol_confirm[i]):
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