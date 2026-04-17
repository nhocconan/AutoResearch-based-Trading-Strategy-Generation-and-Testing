#!/usr/bin/env python3
"""
12h_Williams_Alligator_ElderRay_Trend
Strategy: 12h Williams Alligator for trend direction + Elder Ray for entry/exit.
Long: Alligator bullish (green > red > blue) + Bull Power > 0 and rising
Short: Alligator bearish (red > green > blue) + Bear Power < 0 and falling
Exit: Alligator flips or power crosses zero
Position size: 0.25
Designed to capture trends with confirmation from bull/bear power.
Timeframe: 12h
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
    
    # Williams Alligator: SMAs of median price
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values  # 13-period smoothed by 8
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values   # 8-period smoothed by 5
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values   # 5-period smoothed by 3
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Momentum of power
    bull_momentum = bull_power - np.roll(bull_power, 1)
    bear_momentum = bear_power - np.roll(bear_power, 1)
    bull_momentum[0] = 0
    bear_momentum[0] = 0
    
    # Alligator alignment
    alligator_bullish = (lips > teeth) & (teeth > jaw)
    alligator_bearish = (jaw > teeth) & (teeth > lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # max of all periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_momentum[i]) or np.isnan(bear_momentum[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Alligator bullish + Bull Power > 0 and rising
            if alligator_bullish[i] and bull_power[i] > 0 and bull_momentum[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + Bear Power < 0 and falling
            elif alligator_bearish[i] and bear_power[i] < 0 and bear_momentum[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator flips bearish OR Bull Power crosses zero
            if not alligator_bullish[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator flips bullish OR Bear Power crosses zero
            if not alligator_bearish[i] or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_ElderRay_Trend"
timeframe = "12h"
leverage = 1.0