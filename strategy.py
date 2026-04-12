#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_alligator_elderray_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Elder Ray Power (13-period EMA)
    close_1d = df_1d['close'].values
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Using current high vs 13 EMA
    bear_power = low - ema13   # Using current low vs 13 EMA
    
    # Align Elder Ray to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Alligator lines (13, 8, 5 SMAs with future shifts)
    # Jaw (13-period SMMA shifted 8 bars)
    sma13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(sma13, 8)  # Shift forward 8 bars
    jaw[:8] = np.nan
    
    # Teeth (8-period SMMA shifted 5 bars)
    sma8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(sma8, 5)  # Shift forward 5 bars
    teeth[:5] = np.nan
    
    # Lips (5-period SMMA shifted 3 bars)
    sma5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(sma5, 3)  # Shift forward 3 bars
    lips[:3] = np.nan
    
    # Align Alligator to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Elder Ray conditions
        bull_power_pos = bull_power_aligned[i] > 0
        bear_power_neg = bear_power_aligned[i] < 0
        
        # Alligator alignment (all three lines in order)
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish alignment: Jaw > Teeth > Lips
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # Long signal: Bullish Elder Ray + Bullish Alligator alignment
        long_signal = bull_power_pos and bear_power_neg and bullish_alignment
        # Short signal: Bearish Elder Ray + Bearish Alligator alignment
        short_signal = (not bull_power_pos) and (not bear_power_neg) and bearish_alignment
        
        # Exit conditions: Elder Ray divergence or Alligator crossing
        exit_long = (bull_power_aligned[i] <= 0) or (lips_aligned[i] < jaw_aligned[i])
        exit_short = (bear_power_aligned[i] >= 0) or (lips_aligned[i] > jaw_aligned[i])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals