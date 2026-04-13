#!/usr/bin/env python3
"""
4h_1d_Camarilla_Range_Breakout
Hypothesis: In range-bound markets (Choppiness Index > 61.8), price reverts to mean at S3/R3 levels. 
In trending markets (Choppiness Index < 38.2), breakouts of R3/S3 continue the trend. 
Uses 1d Camarilla levels for support/resistance and 1d Choppiness Index for regime filter.
Works in both bull and bear markets by adapting to regime. Target: 25-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    range_val = high - low
    range_val = np.where(range_val == 0, 1e-10, range_val)
    C = close
    H = high
    L = low
    R1 = C + ((H - L) * 1.0833)
    R2 = C + ((H - L) * 1.1666)
    R3 = C + ((H - L) * 1.2500)
    R4 = C + ((H - L) * 1.5000)
    S1 = C - ((H - L) * 1.0833)
    S2 = C - ((H - L) * 1.1666)
    S3 = C - ((H - L) * 1.2500)
    S4 = C - ((H - L) * 1.5000)
    return R1, R2, R3, R4, S1, S2, S3, S4

def calculate_choppiness(high, low, close, window=14):
    """Calculate Choppiness Index."""
    atr = []
    for i in range(len(high)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr.append(tr)
    
    atr = np.array(atr)
    atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum()
    
    highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
    lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
    range_max_min = highest_high - lowest_low
    
    chop = 100 * np.log10(atr_sum / range_max_min) / np.log10(window)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d indicators
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, window=14)
    
    # Align all data to 4h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        chop_value = chop_1d_aligned[i]
        
        # Regime-based logic
        if chop_value > 61.8:  # Range market - mean reversion
            # Buy near S3, sell near R3
            long_condition = (low[i] <= S3_1d_aligned[i]) and volume_expansion[i]
            short_condition = (high[i] >= R3_1d_aligned[i]) and volume_expansion[i]
            
            # Exit at opposite levels
            long_exit = (high[i] >= (S3_1d_aligned[i] + (R3_1d_aligned[i] - S3_1d_aligned[i]) * 0.5))
            short_exit = (low[i] <= (R3_1d_aligned[i] - (R3_1d_aligned[i] - S3_1d_aligned[i]) * 0.5))
            
        else:  # Trending market - breakout continuation
            # Buy breakouts above R3, sell breakdowns below S3
            long_condition = (high[i] > R3_1d_aligned[i]) and volume_expansion[i]
            short_condition = (low[i] < S3_1d_aligned[i]) and volume_expansion[i]
            
            # Exit on re-entry into range
            long_exit = (close[i] < R3_1d_aligned[i])
            short_exit = (close[i] > S3_1d_aligned[i])
        
        # Update position
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif long_exit and position == 1:
            position = 0
            signals[i] = 0.0
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_exit and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Camarilla_Range_Breakout"
timeframe = "4h"
leverage = 1.0