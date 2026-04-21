#!/usr/bin/env python3
"""
6h Williams Alligator + 1d Volume Spike + 1d Price Reversal
Williams Alligator identifies trend alignment (Jaws/Teeth/Lips). Enter when price closes
outside the Alligator's mouth with volume spike (>2x 20-period average). Exit on reversal
signal (price crosses back into mouth) or volatility contraction. Designed for 6h timeframe
to capture medium-term trends with minimal trades (~30-60/year) to reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Williams Alligator and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Williams Alligator: SMAs of median price
    # Median price = (high + low) / 2
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Jaws: 13-period SMMA, shifted 8 bars
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaws = jaws.shift(8)
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    # Align Alligator lines
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    
    # Volume spike: current volume / 20-period average
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean()
    vol_ratio = df_1d['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        jaw_val = jaws_aligned[i]
        tooth_val = teeth_aligned[i]
        lip_val = lips_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 2.0  # Volume must be 2x average for confirmation
        
        if position == 0:
            # Enter long: price above all three lines (bullish alignment) + volume spike
            if (price_close > jaw_val and price_close > tooth_val and price_close > lip_val and
                vol_ratio > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price below all three lines (bearish alignment) + volume spike
            elif (price_close < jaw_val and price_close < tooth_val and price_close < lip_val and
                  vol_ratio > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back into the Alligator's mouth (between jaws and lips)
            # or volume drops significantly (loss of momentum)
            if position == 1 and (price_close < jaw_val or price_close > lip_val):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > jaw_val or price_close < lip_val):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dVolumeSpike_Reversal"
timeframe = "6h"
leverage = 1.0