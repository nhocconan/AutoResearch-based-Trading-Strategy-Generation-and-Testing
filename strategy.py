#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike for trend following.
Long when Alligator jaws-teeth-lips aligned bullish (jaws>teeth>lips) AND Elder Bull Power > 0 AND volume > 1.5x 20-period MA.
Short when Alligator aligned bearish (jaws<teeth<lips) AND Elder Bear Power < 0 AND volume > 1.5x 20-period MA.
Exit when Alligator alignment breaks or Elder Power reverses.
Uses 1d HTF for Elder Power calculation to reduce noise and avoid whipsaws.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Williams Alligator identifies trend alignment, Elder Power measures bull/bear strength, volume confirms momentum.
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
    
    # Calculate Williams Alligator (SMAs with specific periods)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    # Using EMA as proxy for SMMA for simplicity (common approximation)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5)
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Calculate 1d Elder Power (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Elder Bull Power = High - EMA(13)
    # Elder Bear Power = Low - EMA(13)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align to LTF
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate volume MA (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 8, 5, 13, 20) + 8  # Alligator max shift + buffers
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_vals[i]) or np.isnan(teeth_vals[i]) or np.isnan(lips_vals[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator alignment
        jaw_val = jaw_vals[i]
        teeth_val = teeth_vals[i]
        lips_val = lips_vals[i]
        
        alligator_bullish = jaw_val > teeth_val > lips_val
        alligator_bearish = jaw_val < teeth_val < lips_val
        
        # Elder Power signals
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        
        # Volume filter
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Alligator bullish AND Bull Power > 0 AND volume filter
            if alligator_bullish and bull_power > 0 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power < 0 AND volume filter
            elif alligator_bearish and bear_power < 0 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator alignment breaks OR Bull Power <= 0
                if not alligator_bullish or bull_power <= 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator alignment breaks OR Bear Power >= 0
                if not alligator_bearish or bear_power >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Williams_Alligator_ElderRay_VolumeSpike"
timeframe = "4h"
leverage = 1.0