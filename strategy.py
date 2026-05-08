#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d Elder Ray (Bull/Bear Power) for trend confirmation.
# The Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) identifies trend absence when lines are intertwined.
# Trade only when Alligator lines are separated (trending) AND Elder Ray confirms direction.
# Long: Lips > Teeth > Jaw AND Bull Power > 0 (close > EMA13)
# Short: Lips < Teeth < Jaw AND Bear Power < 0 (close < EMA13)
# Exit when Alligator lines re-intertwine (trend weakening).
# Uses 6h timeframe with 1d Elder Ray filter to reduce false signals.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency.

name = "6h_WilliamsAlligator_1dElderRay"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator on 6h data
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Daily data for Elder Ray (Bull/Bear Power)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 13:
        return np.zeros(n)
    
    close_d = df_d['close'].values
    ema13_d = pd.Series(close_d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_d = df_d['high'].values - ema13_d
    bear_power_d = df_d['low'].values - ema13_d
    
    # Align Elder Ray to 6h timeframe
    bull_power = align_htf_to_ltf(prices, df_d, bull_power_d)
    bear_power = align_htf_to_ltf(prices, df_d, bear_power_d)
    
    # Alligator trend detection: lines separated (not intertwined)
    # Jaw-Teeth-Teeth-Lips order determines trend direction
    jaw_above_teeth = jaw > teeth
    teeth_above_lips = teeth > lips
    jaw_below_teeth = jaw < teeth
    teeth_below_lips = teeth < lips
    
    # Strong uptrend: Jaw < Teeth < Lips (lines separated, bullish alignment)
    strong_uptrend = jaw_below_teeth & teeth_below_lips
    # Strong downtrend: Jaw > Teeth > Lips (lines separated, bearish alignment)
    strong_downtrend = jaw_above_teeth & teeth_above_lips
    # Weak/no trend: lines intertwined (not separated)
    weak_trend = ~(strong_uptrend | strong_downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period, teeth_period, lips_period, 13)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: strong uptrend AND bullish Elder Ray
            long_cond = strong_uptrend[i] and (bull_power[i] > 0)
            # Short conditions: strong downtrend AND bearish Elder Ray
            short_cond = strong_downtrend[i] and (bear_power[i] < 0)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend weakening (lines intertwining) OR bearish power
            if weak_trend[i] or (bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend weakening OR bullish power
            if weak_trend[i] or (bull_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals