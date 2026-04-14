#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour strategy combining 1-day Williams Alligator trend filter with 4-hour price action.
# The Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends.
# When all three lines are aligned (Jaw > Teeth > Lips for uptrend, Jaw < Teeth < Lips for downtrend),
# we take trades in the direction of the trend on 4-hour timeframe.
# Entry: 4-hour close crosses the 8-period SMA in direction of 1-day Alligator alignment.
# Exit: When the 1-day Alligator lines become misaligned (trend weakness) or opposite alignment occurs.
# This captures strong trends while avoiding choppy markets, working in both bull and bear phases.
# Position size: 0.25 (25%) to balance risk and return.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator on 1d
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    if len(df_1d) < jaw_period:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Jaw (13-period SMA)
    jaw = pd.Series(close_1d).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    # Teeth (8-period SMA)
    teeth = pd.Series(close_1d).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    # Lips (5-period SMA)
    lips = pd.Series(close_1d).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 4-hour indicators
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(jaw_period, 8)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or
            np.isnan(sma_8[i])):
            signals[i] = 0.0
            continue
        
        # Check Alligator alignment
        bullish_alignment = (jaw_aligned[i] > teeth_aligned[i] and 
                            teeth_aligned[i] > lips_aligned[i])
        bearish_alignment = (jaw_aligned[i] < teeth_aligned[i] and 
                            teeth_aligned[i] < lips_aligned[i])
        
        if position == 0:
            if bullish_alignment:
                # Bullish trend: go long when price crosses above 8 SMA
                if close[i] > sma_8[i] and (i == start or close[i-1] <= sma_8[i-1]):
                    position = 1
                    signals[i] = position_size
            elif bearish_alignment:
                # Bearish trend: go short when price crosses below 8 SMA
                if close[i] < sma_8[i] and (i == start or close[i-1] >= sma_8[i-1]):
                    position = -1
                    signals[i] = -position_size
        elif position == 1:
            # Exit long: when bullish alignment breaks or bearish alignment appears
            if not bullish_alignment or bearish_alignment:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: when bearish alignment breaks or bullish alignment appears
            if not bearish_alignment or bullish_alignment:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsAlligator_TrendFollow_v1"
timeframe = "4h"
leverage = 1.0