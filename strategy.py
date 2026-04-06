#!/usr/bin/env python3
"""
Experiment #11799: 6h Williams Alligator + Elder Ray Momentum
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
Elder Ray (Bull Power/Bear Power) measures momentum behind the trend.
Combined, they filter false breakouts in chop while capturing sustained moves.
Works in bull (Alligator aligned up, Bull Power positive) and bear (aligned down, Bear Power negative).
Target: 50-150 trades over 4 years. Uses 60-period SMAs for smoother signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11799_6h_williams_alligator_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 60  # Base period for smoothed SMAs
JAW_OFFSET = 3         # Jaw: SMMA(close, 13*8) -> ~624 periods, but we use scaled
TEETH_OFFSET = 2       # Teeth: SMMA(close, 8*8) -> ~384
LIPS_OFFSET = 1        # Lips: SMMA(close, 5*8) -> ~240
ELDER_RAY_PERIOD = 13  # Standard for Elder Ray
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def smma(source, period):
    """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
    return pd.Series(source).ewm(alpha=1/period, adjust=False).mean().values

def calculate_alligator(close, period):
    """Williams Alligator: Jaw (blue), Teeth (red), Lips (green)"""
    jaw = smma(close, period * 8)  # 13*8 = 104 smoothed
    teeth = smma(close, period * 5) # 8*8 = 64 smoothed
    lips = smma(close, period * 3)  # 5*8 = 40 smoothed
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, period):
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def calculate_atr(high, low, close, period):
    """ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Calculate indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(close, ALLIGATOR_PERIOD)
    
    # Elder Ray
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # ATR for stops
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Warmup: need enough data for Alligator
    start = int(ALLIGATOR_PERIOD * 8) + 10
    
    for i in range(start, n):
        # Skip if indicators not ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, reverse for downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray confirmation: Bull Power rising, Bear Power falling
        # Use current vs previous to check momentum
        if i > 0:
            bull_power_rising = bull_power[i] > bull_power[i-1]
            bear_power_falling = bear_power[i] < bear_power[i-1]
        else:
            bull_power_rising = False
            bear_power_falling = False
        
        # Entry conditions
        long_entry = alligator_long and bull_power_rising and (bull_power[i] > 0)
        short_entry = alligator_short and bear_power_falling and (bear_power[i] < 0)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals