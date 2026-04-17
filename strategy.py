#!/usr/bin/env python3
"""
4h_Alligator_ElderRay_Direction_Plus_VolumeSpike
Hypothesis: On 4h timeframe, use Williams Alligator (13,8,5 SMAs) and Elder Ray (bull/bear power from EMA13) to determine trend direction. Enter long when bull power > 0 and price > Alligator teeth (red line) with volume spike (>1.5x 20-period median volume). Enter short when bear power < 0 and price < Alligator teeth with volume spike. Exit on opposite signal. Designed for low trade frequency (<50/year) to work in both bull and bear markets by requiring trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs"""
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    return jaw, teeth, lips

def calculate_elder_ray(close, high, low, ema_period=13):
    """Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13"""
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power, ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Williams Alligator and Elder Ray on 4h (same timeframe) ===
    jaw, teeth, lips = calculate_alligator(high, low, close)
    bull_power, bear_power, ema13 = calculate_elder_ray(close, high, low)
    
    # === Volume confirmation (20-period median volume) ===
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_median[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike: current volume > 1.5x median volume
        vol_spike = volume[i] > 1.5 * vol_median[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bull power > 0 (bullish energy), price > teeth (above Alligator's teeth), volume spike
            if bull_power[i] > 0 and close[i] > teeth[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                continue
            # Short: bear power < 0 (bearish energy), price < teeth (below Alligator's teeth), volume spike
            elif bear_power[i] < 0 and close[i] < teeth[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal
        elif position == 1:
            # Exit long when bear power < 0 and price < teeth (bearish takeover)
            if bear_power[i] < 0 and close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when bull power > 0 and price > teeth (bullish takeover)
            if bull_power[i] > 0 and close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Alligator_ElderRay_Direction_Plus_VolumeSpike"
timeframe = "4h"
leverage = 1.0