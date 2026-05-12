#!/usr/bin/env python3
# 6h_WilliamsAlligator_ElderRay_TrendFollow
# Hypothesis: On 6h timeframe, use Williams Alligator (3 SMAs) for trend direction and Elder Ray (bull/bear power) for momentum confirmation.
# Enter long when price > Alligator teeth (middle SMA) and bull power > 0 and rising.
# Enter short when price < Alligator teeth and bear power < 0 and falling.
# Exit when price crosses Alligator teeth or Elder Ray momentum fades.
# Uses Williams Alligator (13,8,5 SMAs) and Elder Ray (EMA13) to capture trends while avoiding whipsaws in both bull and bear markets.
# Targets 15-25 trades/year for low fee drag.

name = "6h_WilliamsAlligator_ElderRay_TrendFollow"
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
    
    # Calculate Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # Using SMA as proxy for SMMA (Smoothed Moving Average) for simplicity
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # Alligator Jaw
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # Alligator Teeth
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # Alligator Lips
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Smooth Elder Ray for momentum confirmation
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        teeth_val = teeth[i]
        bull_val = bull_power_smooth[i]
        bear_val = bear_power_smooth[i]
        
        if position == 0:
            # LONG: Price > Alligator teeth AND bull power > 0 AND bull power rising
            if close[i] > teeth_val and bull_val > 0 and bull_val > bull_power_smooth[i-1]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price < Alligator teeth AND bear power < 0 AND bear power falling (more negative)
            elif close[i] < teeth_val and bear_val < 0 and bear_val < bear_power_smooth[i-1]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < Alligator teeth OR bull power <= 0 OR bull power falling
            if close[i] < teeth_val or bull_val <= 0 or bull_val < bull_power_smooth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > Alligator teeth OR bear power >= 0 OR bear power rising (less negative)
            if close[i] > teeth_val or bear_val >= 0 or bear_val > bear_power_smooth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals