#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h Williams Alligator regime filter
# - Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength
# - Williams Alligator (Jaw=TEETH=13, Teeth=8, Lips=5 SMAs) identifies trend vs range
# - Long when Bull Power > 0 AND Bear Power < 0 AND Alligator aligned bullish (Lips > Teeth > Jaw)
# - Short when Bear Power > 0 AND Bull Power < 0 AND Alligator aligned bearish (Lips < Teeth < Jaw)
# - Uses 12h HTF for Alligator regime to avoid 6h whipsaw, 6h for Elder Ray timing
# - Discrete position sizing ±0.25 limits drawdown and reduces fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 6h
# - Works in bull markets (strong Elder Ray + bullish Alligator) and bear markets (strong Elder Ray + bearish Alligator)

name = "6h_12h_elder_ray_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for Williams Alligator regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h Williams Alligator SMAs
    close_12h = df_12h['close'].values
    jaw_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values  # Jaw (13-period)
    teeth_12h = pd.Series(close_12h).ewm(span=8, adjust=False, min_periods=8).mean().values    # Teeth (8-period)
    lips_12h = pd.Series(close_12h).ewm(span=5, adjust=False, min_periods=5).mean().values     # Lips (5-period)
    
    # Align 12h Alligator to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Pre-compute 6h Elder Ray components
    # EMA13 for Elder Ray calculation
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13_6h  # Bull Power = High - EMA13
    bear_power = ema13_6h - low   # Bear Power = EMA13 - Low
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment conditions
        alligator_bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        alligator_bearish = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Elder Ray conditions
        strong_bull = bull_power[i] > 0 and bear_power[i] < 0  # Bull Power positive, Bear Power negative
        strong_bear = bear_power[i] > 0 and bull_power[i] < 0  # Bear Power positive, Bull Power negative
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Strong bullish Elder Ray + bullish Alligator alignment
        if strong_bull and alligator_bullish:
            enter_long = True
        
        # Short: Strong bearish Elder Ray + bearish Alligator alignment
        if strong_bear and alligator_bearish:
            enter_short = True
        
        # Exit conditions: opposite Elder Ray signal or Alligator alignment breakdown
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bearish Elder Ray appears OR Alligator loses bullish alignment
            exit_long = strong_bear or (not alligator_bullish)
        elif position == -1:
            # Exit short if bullish Elder Ray appears OR Alligator loses bearish alignment
            exit_short = strong_bull or (not alligator_bearish)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals