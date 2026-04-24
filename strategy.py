#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike for trend following.
- Primary timeframe: 4h for execution.
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3).
  Long when Lips > Teeth > Jaw and all rising; Short when Lips < Teeth < Jaw and all falling.
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
  Confirm long when Bull Power > 0 and rising; short when Bear Power > 0 and rising.
- Volume confirmation: current volume > 2.0 x 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying strong uptrends, in bear via selling strong downtrends.
- Uses SMMA (Smoothed Moving Average) for Alligator lines.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=float)
    smma_arr = np.full_like(source, np.nan, dtype=float)
    if len(source) < length:
        return smma_arr
    # First value is SMA
    smma_arr[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CLOSE) / length
    for i in range(length, len(source)):
        smma_arr[i] = (smma_arr[i-1] * (length-1) + source[i]) / length
    return smma_arr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator and Elder Ray (same timeframe, no alignment needed)
    # But we still use get_htf_data for consistency and to follow MTF rules
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need enough for SMMA(13) + shifts
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Williams Alligator
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw_raw = smma(close_4h, 13)
    jaw = np.roll(jaw_raw, 8)  # Shift right by 8 (future values move to past)
    jaw[:8] = np.nan  # First 8 values become invalid after shift
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth_raw = smma(close_4h, 8)
    teeth = np.roll(teeth_raw, 5)  # Shift right by 5
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips_raw = smma(close_4h, 5)
    lips = np.roll(lips_raw, 3)  # Shift right by 3
    lips[:3] = np.nan
    
    # Elder Ray (using 13-period EMA)
    ema_13 = pd.Series(close_4h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_4h - ema_13
    bear_power = ema_13 - low_4h
    
    # Volume confirmation: current volume > 2.0 x 20-period volume MA
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_4h > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(21, 20)  # Alligator longest jaw (13+8=21) + volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check Alligator alignment and direction
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Check if Alligator lines are rising (current > previous)
        lips_rising = lips[i] > lips[i-1]
        teeth_rising = teeth[i] > teeth[i-1]
        jaw_rising = jaw[i] > jaw[i-1]
        lips_falling = lips[i] < lips[i-1]
        teeth_falling = teeth[i] < teeth[i-1]
        jaw_falling = jaw[i] < jaw[i-1]
        
        if position == 0:
            # Long condition: Lips > Teeth > Jaw and all rising + Bull Power > 0 and rising + Volume spike
            if (lips_above_teeth and teeth_above_jaw and lips_rising and teeth_rising and jaw_rising and
                bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short condition: Lips < Teeth < Jaw and all falling + Bear Power > 0 and rising + Volume spike
            elif (lips_below_teeth and teeth_below_jaw and lips_falling and teeth_falling and jaw_falling and
                  bear_power[i] > 0 and bear_power[i] > bear_power[i-1] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator loses alignment (Lips < Teeth or Teeth < Jaw) or Bull Power turns negative
            if not (lips_above_teeth and teeth_above_jaw) or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator loses alignment (Lips > Teeth or Teeth > Jaw) or Bear Power turns negative
            if not (lips_below_teeth and teeth_below_jaw) or bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0