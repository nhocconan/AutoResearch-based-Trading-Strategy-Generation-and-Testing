#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray power confluence.
Long when: Alligator bullish (jaw < teeth < lips) AND Elder Bull Power > 0 AND price > EMA13.
Short when: Alligator bearish (jaw > teeth > lips) AND Elder Bear Power < 0 AND price < EMA13.
Uses 12h HTF for Alligator alignment (avoids whipsaws). Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # Calculate 12h Williams Alligator (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3) smoothed median price
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 6h EMA13 and Elder Ray power (LTF)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_prev = np.roll(ema_13, 1)
    ema_13_prev[0] = np.nan
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 8, 5) + 8  # Alligator longest shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema13_val = ema_13[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Alligator alignment
        alligator_bullish = jaw_val < teeth_val < lips_val
        alligator_bearish = jaw_val > teeth_val > lips_val
        
        if position == 0:
            # Long: Alligator bullish AND Bull Power > 0 AND price > EMA13
            if alligator_bullish and bull_val > 0 and price > ema13_val:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power < 0 AND price < EMA13
            elif alligator_bearish and bear_val < 0 and price < ema13_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator turns bearish OR Bear Power > 0
                if alligator_bearish or bear_val > 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator turns bullish OR Bull Power < 0
                if alligator_bullish or bull_val < 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsAlligator_ElderRay_Power_Confluence"
timeframe = "6h"
leverage = 1.0