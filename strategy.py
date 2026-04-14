#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator system with 1-day Elder Ray Index for trend confirmation.
# The Williams Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets.
# The Elder Ray Index (Bull Power/Bear Power) from 1-day data confirms institutional strength.
# Entry occurs when Lips cross above Teeth (bullish) or below Teeth (bearish) with Elder Ray confirmation.
# Exit occurs when Lips cross back through Teeth in the opposite direction.
# This combination filters false signals in ranging markets and captures strong trends in both bull and bear markets.
# Target: 20-35 trades per year per symbol (80-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1-day data ONCE for Elder Ray Index
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator parameters (13, 8, 5) with future shifts (8, 5, 3)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    if len(close) < jaw_period:
        return np.zeros(n)
    
    # Calculate Alligator lines
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(5).values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().shift(3).values
    
    # Elder Ray Index from 1-day data
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray
    ema_13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Align Elder Ray to 4h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(jaw_period + 8, teeth_period + 5, lips_period + 3, 13)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator signals
        lips_above_teeth = lips[i] > teeth[i]
        lips_below_teeth = lips[i] < teeth[i]
        
        # Elder Ray confirmation: Bull Power > 0 and Bear Power < 0 for strong trend
        strong_bullish = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        strong_bearish = bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0
        
        if position == 0:
            # Enter long: Lips cross above Teeth + strong bullish Elder Ray
            if lips_above_teeth and strong_bullish:
                position = 1
                signals[i] = position_size
            # Enter short: Lips cross below Teeth + strong bearish Elder Ray
            elif lips_below_teeth and strong_bearish:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Lips cross back below Teeth (contrary signal)
            if lips_below_teeth:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Lips cross back above Teeth (contrary signal)
            if lips_above_teeth:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Alligator_ElderRay_Trend_v1"
timeframe = "4h"
leverage = 1.0