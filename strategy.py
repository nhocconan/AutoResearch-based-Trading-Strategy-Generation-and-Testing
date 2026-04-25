#!/usr/bin/env python3
"""
12h Williams Alligator + Elder Ray Power with 1w Trend Filter
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trend absence (alligator sleeping) vs presence (awake). Elder Ray Power (Bull/Bear Power) measures momentum strength. Using 1w EMA34 as higher-timeframe trend filter ensures alignment with weekly trend, reducing false signals in choppy markets. Works in bull markets (long when Bull Power > 0 and price above teeth) and bear markets (short when Bear Power < 0 and price below teeth) by requiring trend alignment. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-30 trades/year on 12h.
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
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator (13, 8, 5 periods with shifts 8, 5, 3)
    # Jaw (Blue Line): 13-period SMMA shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (Red Line): 8-period SMMA shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (Green Line): 5-period SMMA shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray Power (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator calculations (13+8=21 max shift)
    start_idx = 21
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        weekly_trend = ema_34_1w_aligned[i]
        
        # Alligator sleeping condition: all lines intertwined (no clear trend)
        # Alligator awake: jaws, teeth, lips are separated and ordered
        # For uptrend: Lips > Teeth > Jaw
        # For downtrend: Jaw > Teeth > Lips
        alligator_sleeping = (
            (curr_lips >= curr_jaw and curr_lips <= curr_teeth) or
            (curr_teeth >= curr_jaw and curr_teeth <= curr_lips) or
            (curr_jaw >= curr_lips and curr_jaw <= curr_teeth)
        )
        alligator_awake_uptrend = (curr_lips > curr_teeth) and (curr_teeth > curr_jaw)
        alligator_awake_downtrend = (curr_jaw > curr_teeth) and (curr_teeth > curr_lips)
        
        if position == 0:
            # Look for entry signals
            # Long: Alligator awake (uptrend) AND Bull Power > 0 AND price > weekly EMA34 (uptrend)
            long_entry = alligator_awake_uptrend and (curr_bull > 0) and (curr_close > weekly_trend)
            # Short: Alligator awake (downtrend) AND Bear Power < 0 AND price < weekly EMA34 (downtrend)
            short_entry = alligator_awake_downtrend and (curr_bear < 0) and (curr_close < weekly_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator sleeping OR Bear Power > 0 (momentum shift) OR price < weekly EMA34 (trend change)
            if alligator_sleeping or (curr_bear > 0) or (curr_close < weekly_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator sleeping OR Bull Power < 0 (momentum shift) OR price > weekly EMA34 (trend change)
            if alligator_sleeping or (curr_bull < 0) or (curr_close > weekly_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_Power_1wEMA34_Trend"
timeframe = "12h"
leverage = 1.0