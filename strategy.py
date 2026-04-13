#!/usr/bin/env python3
"""
12h_1d_Williams_Alligator_Trend_Follow
Hypothesis: Use Williams Alligator (3 SMAs) on 1d timeframe to identify trend direction on 12h chart.
Go long when price > Alligator's Jaw (13-period SMA) with teeth and lips aligned bullish.
Go short when price < Jaw with teeth and lips aligned bearish.
Williams Alligator excels in trending markets and avoids whipsaws in ranging markets.
Works in bull markets (strong uptrends) and bear markets (strong downtrends).
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs."""
    # Typical price
    typical = (high + low + close) / 3.0
    
    # Jaw: 13-period SMA, shifted 8 bars forward
    jaw = pd.Series(typical).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth = pd.Series(typical).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Lips: 5-period SMA, shifted 3 bars forward
    lips = pd.Series(typical).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    return jaw.values, teeth.values, lips.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams Alligator
    jaw_1d, teeth_1d, lips_1d = calculate_alligator(high_1d, low_1d, close_1d)
    
    # Align Alligator lines to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw and price > Jaw
        bullish_aligned = (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]) and (close[i] > jaw_1d_aligned[i])
        
        # Bearish alignment: Lips < Teeth < Jaw and price < Jaw
        bearish_aligned = (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]) and (close[i] < jaw_1d_aligned[i])
        
        if bullish_aligned and position != 1:
            position = 1
            signals[i] = position_size
        elif bearish_aligned and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_Williams_Alligator_Trend_Follow"
timeframe = "12h"
leverage = 1.0