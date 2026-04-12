#!/usr/bin/env python3
"""
6h_12h_alligator_elderray_v1
Combines Williams Alligator trend direction from 12h with Elder Ray (Bull/Bear Power) on 6h.
Long when: 12h Alligator bullish (jaw < teeth < lips) AND 6h Bull Power > 0 AND Bear Power < 0
Short when: 12h Alligator bearish (jaw > teeth > lips) AND 6h Bear Power > 0 AND Bull Power < 0
Exit when Alligator reverses or Elder Power signals weaken.
Uses 13,8,5 SMAs for Alligator and 13-period EMA for Elder Power.
Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
Works in trending markets by following Alligator alignment, avoids whipsaws via Elder Ray confirmation.
"""

name = "6h_12h_alligator_elderray_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Williams Alligator on 12h: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    # Alligator alignment: bullish when jaw < teeth < lips, bearish when jaw > teeth > lips
    alligator_bullish = (jaw < teeth) & (teeth < lips)
    alligator_bearish = (jaw > teeth) & (teeth > lips)
    
    # Align Alligator signals to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    alligator_bullish_aligned = align_htf_to_ltf(prices, df_12h, alligator_bullish)
    alligator_bearish_aligned = align_htf_to_ltf(prices, df_12h, alligator_bearish)
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(alligator_bullish_aligned[i]) or np.isnan(alligator_bearish_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: 12h Alligator bullish AND 6h Bull Power > 0 AND Bear Power < 0
        if alligator_bullish_aligned[i] and bull_power[i] > 0 and bear_power[i] < 0 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: 12h Alligator bearish AND 6h Bear Power > 0 AND Bull Power < 0
        elif alligator_bearish_aligned[i] and bear_power[i] > 0 and bull_power[i] < 0 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (not alligator_bullish_aligned[i] or bull_power[i] <= 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not alligator_bearish_aligned[i] or bear_power[i] <= 0):
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