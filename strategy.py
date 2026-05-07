#!/usr/bin/env python3
"""
1D_Williams_Alligator_Trend_Filter_v1
Hypothesis: Use weekly Williams Alligator (3 SMAs: Jaw-Teeth-Lips) for trend direction and daily price action for entries.
Long when price > Alligator Teeth and price > previous day's high (breakout); 
Short when price < Alligator Teeth and price < previous day's low (breakdown).
Volume confirmation: current volume > 1.3x 20-day average volume.
Williams Alligator smooths noise and identifies strong trends, reducing whipsaws in both bull and bear markets.
"""
name = "1D_Williams_Alligator_Trend_Filter_v1"
timeframe = "1d"
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
    
    # Get weekly data for Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator (3 SMAs: 13, 8, 5 periods with future shifts)
    close_1w = pd.Series(df_1w['close'])
    # Jaw: 13-period SMA, shifted 8 bars forward
    jaw = close_1w.rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth = close_1w.rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: 5-period SMA, shifted 3 bars forward
    lips = close_1w.rolling(window=5, min_periods=5).mean().shift(3)
    
    # Align to daily timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips.values)
    
    # Volume filter: current volume > 1.3 * 20-day average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13+8)  # Ensure sufficient warmup for Alligator
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(teeth_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Teeth and breakout above previous day's high
            if (close[i] > teeth_aligned[i] and 
                high[i] > high[i-1] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < Teeth and breakdown below previous day's low
            elif (close[i] < teeth_aligned[i] and 
                  low[i] < low[i-1] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses Teeth in opposite direction
            if position == 1 and close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals