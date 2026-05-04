#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike
# Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) defines trend: 
#   Bullish: Lips > Teeth > Jaw | Bearish: Lips < Teeth < Jaw
# Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures trend strength
# Volume spike confirms institutional participation
# Works in bull/bear: Alligator filters whipsaws, Elder Ray ensures momentum, Volume avoids low-liquidity traps
# Target: 20-40 trades/year via strict confluence

name = "4h_WilliamsAlligator_ElderRay_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    median_1d = (high_1d + low_1d) / 2
    jaw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align to 4h (wait for completed 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish Alligator alignment AND Bull Power > 0 AND Volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                bull_power_aligned[i] > 0 and 
                volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = 0.30
                position = 1
            # Short: Bearish Alligator alignment AND Bear Power < 0 AND Volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  bear_power_aligned[i] < 0 and 
                  volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bull Power <= 0
            if (lips_aligned[i] <= teeth_aligned[i] or 
                bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bear Power >= 0
            if (lips_aligned[i] >= teeth_aligned[i] or 
                bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals