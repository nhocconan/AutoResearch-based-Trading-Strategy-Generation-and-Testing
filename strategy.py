#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray (Bull/Bear Power) combination
# Uses Williams Alligator (JAW=TEETH=LIPS) from 6h chart to define trend regime and avoid whipsaw,
# combined with 1d Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) for momentum confirmation.
# Designed for 12-35 trades/year (~50-140 total over 4 years) to minimize fee drag.
# Alligator acts as trend filter: only trade when price is outside the Alligator's mouth (JAW-LIPS gap).
# Elder Ray provides entry timing: long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
# Works in both bull/bear markets by adapting to Alligator's trend/range detection.

name = "6h_WilliamsAlligator_1dElderRay_Momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Elder Ray calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Align Elder Ray to 6h timeframe (wait for completed 1d bar)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate Williams Alligator from 6h data (JAW=13, TEETH=8, LIPS=5, all SMMA)
    # SMMA calculation using EMA as approximation (standard practice)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values  # JAW (blue)
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values    # TEETH (red)
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values    # LIPS (green)
    
    # Align Alligator components (already on 6h, no HTF alignment needed)
    # But shift by 1 to use only completed bar values
    jaw_shifted = np.roll(jaw, 1)
    teeth_shifted = np.roll(teeth, 1)
    lips_shifted = np.roll(lips, 1)
    jaw_shifted[0] = np.nan
    teeth_shifted[0] = np.nan
    lips_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator condition: price outside mouth (JAW-LIPS gap) indicates trending market
        # Mouth is defined by JAW and LIPS (outer lines)
        alligator_mouth_up = np.maximum(jaw_shifted[i], lips_shifted[i])
        alligator_mouth_down = np.minimum(jaw_shifted[i], lips_shifted[i])
        price_above_mouth = close[i] > alligator_mouth_up
        price_below_mouth = close[i] < alligator_mouth_down
        
        if position == 0:
            # Long conditions: price above Alligator mouth AND Bull Power positive AND rising
            if (price_above_mouth and 
                bull_power_1d_aligned[i] > 0 and 
                bull_power_1d_aligned[i] > bull_power_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price below Alligator mouth AND Bear Power negative AND falling
            elif (price_below_mouth and 
                  bear_power_1d_aligned[i] < 0 and 
                  bear_power_1d_aligned[i] < bear_power_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Alligator mouth OR Bull Power turns negative
            if (close[i] <= alligator_mouth_up and close[i] >= alligator_mouth_down) or bull_power_1d_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Alligator mouth OR Bear Power turns positive
            if (close[i] <= alligator_mouth_up and close[i] >= alligator_mouth_down) or bear_power_1d_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals