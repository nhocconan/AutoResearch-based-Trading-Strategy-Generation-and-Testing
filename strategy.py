#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Alligator_Channel_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === Williams Alligator on Daily ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(close_1d).rolling(window=13, center=False).mean().shift(8).values
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(close_1d).rolling(window=8, center=False).mean().shift(5).values
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(close_1d).rolling(window=5, center=False).mean().shift(3).values
    
    # Align to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === Daily Price Channel (Donchian 20) ===
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        high_20_val = high_20_aligned[i]
        low_20_val = low_20_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or
            np.isnan(high_20_val) or np.isnan(low_20_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish alignment
        # Alligator alignment: Lips < Teeth < Jaw = bearish alignment
        bullish_align = lips_val > teeth_val and teeth_val > jaw_val
        bearish_align = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long: Bullish alignment + price breaks above 20-day high
            if bullish_align and close_val > high_20_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price breaks below 20-day low
            elif bearish_align and close_val < low_20_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator turns bearish OR price retests 20-day low
            if not bullish_align or close_val < low_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator turns bullish OR price retests 20-day high
            if not bearish_align or close_val > high_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals