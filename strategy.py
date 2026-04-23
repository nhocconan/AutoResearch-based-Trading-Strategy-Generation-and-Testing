#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray Power confluence with 1w pivot bias.
Long when: Alligator aligned bullish (jaw < teeth < lips), Bull Power > 0, price above weekly pivot.
Short when: Alligator aligned bearish (jaw > teeth > lips), Bear Power < 0, price below weekly pivot.
Exit when Alligator alignment breaks or power crosses zero.
Uses 1w HTF for pivot bias and 1d for Alligator/Elder Ray to reduce whipsaws.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d Williams Alligator (13,8,5 SMAs shifted)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Jaw: 13-period SMA shifted 8 bars
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA shifted 5 bars
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA shifted 3 bars
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate 1d Elder Ray Power (Bull/Bear)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 1w pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot: (H+L+C)/3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Weekly R1: 2*P - L
    weekly_r1 = 2 * weekly_pivot - df_1w['low']
    # Weekly S1: 2*P - H
    weekly_s1 = 2 * weekly_pivot - df_1w['high']
    
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 8, 5)  # Alligator/Elder Ray lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        pivot = pivot_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        
        # Alligator alignment
        bullish_aligned = jaw < teeth < lips
        bearish_aligned = jaw > teeth > lips
        
        if position == 0:
            # Long: Bullish Alligator + Bull Power > 0 + price above weekly pivot
            if bullish_aligned and bull_power > 0 and price > pivot:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + Bear Power < 0 + price below weekly pivot
            elif bearish_aligned and bear_power < 0 and price < pivot:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator alignment breaks OR Bull Power <= 0
                if not bullish_aligned or bull_power <= 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator alignment breaks OR Bear Power >= 0
                if not bearish_aligned or bear_power >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsAlligator_ElderRay_1wPivot_Bias"
timeframe = "6h"
leverage = 1.0