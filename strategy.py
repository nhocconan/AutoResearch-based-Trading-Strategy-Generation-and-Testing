#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray combo with 1d EMA34 trend filter and volume confirmation.
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) - trend identification
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 - trend strength
- Long: Alligator aligned (Lips > Teeth > Jaw) + Bull Power > 0 + price > 1d EMA34 + volume > 1.5x 20-period avg
- Short: Alligator aligned inverse (Lips < Teeth < Jaw) + Bear Power < 0 + price < 1d EMA34 + volume > 1.5x 20-period avg
- Exit: Opposite Alligator alignment or power crosses zero
- Uses Williams Alligator for trend structure, Elder Ray for momentum confirmation, 1d EMA34 for HTF trend filter
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (Alligator alignment + Bull Power) and bear markets (inverse alignment + Bear Power)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Smoothed Moving Average (SMA with specific periods)
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    jaw_values = jaw.values
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    teeth_values = teeth.values
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    lips_values = lips.values
    
    # Elder Ray Power
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 13+8, 8+5, 5+3)  # Need 34 for EMA34, 20 for volume MA, Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(jaw_values[i]) or
            np.isnan(teeth_values[i]) or
            np.isnan(lips_values[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Williams Alligator alignment
        lips_gt_teeth = lips_values[i] > teeth_values[i]
        teeth_gt_jaw = teeth_values[i] > jaw_values[i]
        lips_lt_teeth = lips_values[i] < teeth_values[i]
        teeth_lt_jaw = teeth_values[i] < jaw_values[i]
        
        bullish_alligator = lips_gt_teeth and teeth_gt_jaw
        bearish_alligator = lips_lt_teeth and teeth_lt_jaw
        
        if position == 0:
            # Long: Bullish Alligator + Bull Power > 0 + price > 1d EMA34 + volume confirmation
            if (bullish_alligator and 
                bull_power[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + Bear Power < 0 + price < 1d EMA34 + volume confirmation
            elif (bearish_alligator and 
                  bear_power[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Alligator alignment OR Bull Power <= 0
            if not bullish_alligator or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator alignment OR Bear Power >= 0
            if not bearish_alligator or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0