#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d Elder Ray Power filter and volume spike.
- Primary timeframe: 12h, HTF: 1d for Elder Ray trend filter
- Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - uses SMAs with future shift
- Long: Lips > Teeth > Jaw (bullish alignment) + Elder Ray Power > 0 (bull power) + volume > 1.5x 20-period avg
- Short: Lips < Teeth < Jaw (bearish alignment) + Elder Ray Power < 0 (bear power) + volume > 1.5x 20-period avg
- Exit: Reverse Alligator alignment (Lips crosses Teeth in opposite direction)
- Uses Alligator's trend-following nature to catch sustained moves in both bull and bear markets
- Elder Ray confirms institutional buying/selling pressure behind the move
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
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
    
    # Volume confirmation: > 1.5x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1d Elder Ray Power for trend filter
    # Elder Ray Power = Close - EMA13 (bull power when >0, bear power when <0)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 on 1d data
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Elder Ray Bull Power = Close - EMA13
    bull_power_1d = close_1d - ema_13_1d
    # Elder Ray Bear Power = EMA13 - Close (negative of bull power)
    bear_power_1d = ema_13_1d - close_1d
    
    # Align 1d Elder Ray to 12h timeframe (values from previous 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 8, 5)  # Need 20 for volume MA, 13 for jaw shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Alligator alignment checks
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment + positive Elder Ray Bull Power + volume spike
            if (bullish_alignment and 
                bull_power_aligned[i] > 0 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + positive Elder Ray Bear Power + volume spike
            elif (bearish_alignment and 
                  bear_power_aligned[i] > 0 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator loses bullish alignment (Lips crosses below Teeth)
            if lips[i] <= teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator loses bearish alignment (Lips crosses above Teeth)
            if lips[i] >= teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dElderRay_Power_VolumeSpike"
timeframe = "12h"
leverage = 1.0