#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Williams Alligator + volume spike confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 measures trend strength
- Williams Alligator (1d): Jaw(13), Teeth(8), Lips(5) SMAs with offsets to identify trend/no trend
- Alligator sleeping (all lines intertwined) = no trend → avoid trading
- Alligator awakening (lines separated) + Elder Ray alignment = high-probability breakout
- Volume spike (>2x 20-period average) confirms institutional participation
- Discrete position size 0.25 limits drawdown in 2022-like crashes
- Target: 12-30 trades/year on 6h (50-120 total over 4 years)
- Works in bull/bear via Alligator trend filter + Elder Ray alignment
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray on 6h: Bull/Bear Power vs EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High minus EMA13
    bear_power = low - ema_13   # Low minus EMA13
    
    # Williams Alligator on 1d (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Alligator lines: Jaw(13), Teeth(8), Lips(5) SMAs
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Apply Alligator offsets (shifted into future)
    jaw = np.roll(jaw, 8)   # Jaw shifted 8 bars
    teeth = np.roll(teeth, 5) # Teeth shifted 5 bars
    lips = np.roll(lips, 3)   # Lips shifted 3 bars
    
    # Align Alligator to 6t
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20)  # Elder Ray EMA13, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator sleeping condition: lines intertwined (no strong trend)
        max_line = np.maximum(jaw_aligned[i], np.maximum(teeth_aligned[i], lips_aligned[i]))
        min_line = np.minimum(jaw_aligned[i], np.minimum(teeth_aligned[i], lips_aligned[i]))
        alligator_sleeping = (max_line - min_line) < (close[i] * 0.001)  # <0.1% of price
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade when Alligator is awakening (trending)
            if not alligator_sleeping and volume_confirm:
                # Long: Bull Power > 0 AND Lips > Jaw (bullish alignment)
                if bull_power[i] > 0 and lips_aligned[i] > jaw_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 AND Lips < Jaw (bearish alignment)
                elif bear_power[i] < 0 and lips_aligned[i] < jaw_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR Alligator starts sleeping
            if bull_power[i] <= 0 or alligator_sleeping:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive OR Alligator starts sleeping
            if bear_power[i] >= 0 or alligator_sleeping:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Alligator_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0