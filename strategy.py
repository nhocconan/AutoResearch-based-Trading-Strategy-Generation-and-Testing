#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray with volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) for trend direction, Elder Ray (Bull/Bear Power) for momentum,
# and volume spike for confirmation. Designed to capture trends in bull markets and reversals in bear markets.
# Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMA of median price (HL/2) with specific periods
    median_price_1d = (high_1d + low_1d) / 2
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(median_price_1d).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(median_price_1d).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(median_price_1d).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Williams Alligator and Elder Ray to 12h
    jaw_12h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_12h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_12h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need volume MA20 and Elder Ray components
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_12h[i]) or 
            np.isnan(teeth_12h[i]) or 
            np.isnan(lips_12h[i]) or 
            np.isnan(bull_power_12h[i]) or 
            np.isnan(bear_power_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Williams Alligator trend: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        lips_above_teeth = lips_12h[i] > teeth_12h[i]
        teeth_above_jaw = teeth_12h[i] > jaw_12h[i]
        lips_below_teeth = lips_12h[i] < teeth_12h[i]
        teeth_below_jaw = teeth_12h[i] < jaw_12h[i]
        
        # Elder Ray momentum: Bull Power > 0 and rising, Bear Power < 0 and falling
        bull_power_positive = bull_power_12h[i] > 0
        bear_power_negative = bear_power_12h[i] < 0
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (uptrend) AND Bull Power > 0 AND volume spike
            if (lips_above_teeth and teeth_above_jaw and bull_power_positive and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (downtrend) AND Bear Power < 0 AND volume spike
            elif (lips_below_teeth and teeth_below_jaw and bear_power_negative and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Lips < Teeth (loss of uptrend momentum) OR Bear Power > 0 (bullish momentum fading)
            if (lips_12h[i] < teeth_12h[i]) or (bear_power_12h[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Lips > Teeth (loss of downtrend momentum) OR Bull Power < 0 (bearish momentum fading)
            if (lips_12h[i] > teeth_12h[i]) or (bull_power_12h[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0