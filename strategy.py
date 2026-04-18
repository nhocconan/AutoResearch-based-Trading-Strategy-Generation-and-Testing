#!/usr/bin/env python3
"""
4h_Alligator_ElderRay_Trend_With_Volume_Confirmation
4h strategy using Williams Alligator for trend direction, Elder Ray for trend strength,
and volume confirmation. Works in both bull and bear markets by following strong trends
with proper filtering.
Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
Entry: Alligator aligned (Lips > Teeth > Jaw for long, reverse for short) +
       Elder Ray confirms trend strength +
       Volume > 1.5x 20-period average
Exit: Opposite Alligator alignment or Elder Ray divergence
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    if length < 1:
        return source
    smma = np.full_like(source, np.nan, dtype=float)
    smma[length-1] = np.mean(source[:length])
    for i in range(length, len(source)):
        smma[i] = (smma[i-1] * (length-1) + source[i]) / length
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (all SMMA)
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Elder Ray components
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Elder Ray confirmation
        bull_strong = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]
        bear_strong = bear_power[i] < 0 and bear_power[i] < bear_power[i-1]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + Elder Ray bull + volume
            if lips_above_teeth and teeth_above_jaw and bull_strong and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + Elder Ray bear + volume
            elif lips_below_teeth and teeth_below_jaw and bear_strong and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks or Elder Ray turns bearish
            if not (lips_above_teeth and teeth_above_jaw) or not bull_strong:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks or Elder Ray turns bullish
            if not (lips_below_teeth and teeth_below_jaw) or not bear_strong:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Alligator_ElderRay_Trend_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0