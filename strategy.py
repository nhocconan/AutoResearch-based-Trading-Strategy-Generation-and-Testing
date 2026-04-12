#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_alligator_v1
# Uses Williams Alligator (Jaw, Teeth, Lips) from 1-day timeframe to identify trend direction and strength.
# Long when Lips > Teeth > Jaw (bullish alignment) and price > 12h EMA26.
# Short when Lips < Teeth < Jaw (bearish alignment) and price < 12h EMA26.
# Uses volume confirmation (volume > 1.3x 20-period average) to filter false signals.
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
# Alligator acts as a trend-following system that works in both bull and bear markets by
# identifying when the market is trending (aligned) vs ranging (intertwined).

name = "12h_1d_alligator_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate SMAs for Alligator (Williams Alligator uses SMAs, not EMAs)
    close_1d = df_1d['close'].values
    # Jaw: 13-period SMA, shifted 8 bars forward
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift forward 8 bars
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift forward 5 bars
    # Lips: 5-period SMA, shifted 3 bars forward
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift forward 3 bars
    
    # Align Alligator lines to 12h timeframe (daily values update after daily bar closes)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate EMA26 on 12h for entry timing
    ema26_12h = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Volume confirmation: volume > 1.3 * 20-period average (12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) 
            or np.isnan(ema26_12h[i])):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Long signal: bullish alignment AND price above 12h EMA26
        if bullish_alignment and close[i] > ema26_12h[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: bearish alignment AND price below 12h EMA26
        elif bearish_alignment and close[i] < ema26_12h[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite alignment or market becomes ranging (intertwined)
        elif not bullish_alignment and position == 1:
            position = 0
            signals[i] = 0.0
        elif not bearish_alignment and position == -1:
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