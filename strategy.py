# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h Williams Alligator + Elder Ray + Volume Spike
- Uses Williams Alligator (Jaw/Teeth/Lips) from 12h to determine trend regime
- Uses Elder Ray (Bull/Bear Power) from 6h for entry signals
- Volume spike confirmation to filter false signals
- Designed for 6h timeframe to work in both bull and bear markets via regime filtering
"""

name = "6h_Alligator_ElderRay_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator (trend regime)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Williams Alligator: SMAs of median price
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values  # 13-bar SMA
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values   # 8-bar SMA
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values    # 5-bar SMA
    
    # Align to 6t timeframe
    jaw_6h = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_12h, teeth)
    lips_6h = align_htf_to_ltf(prices, df_12h, lips)
    
    # Alligator signals: 
    # Bullish: Lips > Teeth > Jaw (green alignment)
    # Bearish: Lips < Teeth < Jaw (red alignment)
    alligator_bull = (lips_6h > teeth_6h) & (teeth_6h > jaw_6h)
    alligator_bear = (lips_6h < teeth_6h) & (teeth_6h < jaw_6h)
    
    # Elder Ray from 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema13.values
    bear_power = low - ema13.values
    
    # Volume filter: current volume > 1.8x 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(alligator_bull[i]) or np.isnan(alligator_bear[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator bullish + Bull Power > 0 + Volume spike
            if alligator_bull[i] and (bull_power[i] > 0) and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + Bear Power < 0 + Volume spike
            elif alligator_bear[i] and (bear_power[i] < 0) and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bull Power turns negative
            if not alligator_bull[i] or (bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bear Power turns positive
            if not alligator_bear[i] or (bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals