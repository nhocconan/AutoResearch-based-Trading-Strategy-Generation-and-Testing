#!/usr/bin/env python3
# Hypothesis: 6h timeframe with daily Williams Alligator and Elder Ray indicator.
# Uses daily Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) for trend direction and filter.
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
# Entry: Bull Power > 0 and Bear Power < 0 with Alligator bullish alignment (Lips > Teeth > Jaw).
# Exit: Opposite condition or loss of Alligator alignment.
# Designed to work in both bull and bear markets by following strong trends with momentum confirmation.
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25.

name = "6h_Alligator_ElderRay_1dTrend_Power"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate daily Alligator components (SMAs)
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # Jaw: 13-period SMA
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # Teeth: 8-period SMA
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values    # Lips: 5-period SMA
    
    # Align Alligator to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate daily Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power = ema13_1d - low_1d   # Bear Power = EMA13 - Low
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Conditions
    # Alligator bullish: Lips > Teeth > Jaw (aligned and trending up)
    alligator_bullish = (lips_aligned > teeth_aligned) & (teeth_aligned > jaw_aligned)
    # Alligator bearish: Jaws > Teeth > Lips (aligned and trending down)
    alligator_bearish = (jaw_aligned > teeth_aligned) & (teeth_aligned > lips_aligned)
    
    # Elder Ray: Bull Power > 0 and Bear Power < 0 indicates strong bullish momentum
    # Bear Power > 0 and Bull Power < 0 indicates strong bearish momentum
    strong_bullish = (bull_power_aligned > 0) & (bear_power_aligned < 0)
    strong_bearish = (bear_power_aligned > 0) & (bull_power_aligned < 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator bullish + strong bullish momentum
            if alligator_bullish[i] and strong_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + strong bearish momentum
            elif alligator_bearish[i] and strong_bearish[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: loss of bullish alignment or momentum
            if not (alligator_bullish[i] and strong_bullish[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: loss of bearish alignment or momentum
            if not (alligator_bearish[i] and strong_bearish[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals