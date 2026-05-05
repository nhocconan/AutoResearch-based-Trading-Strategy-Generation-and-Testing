#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d Elder Ray Power filter and volume confirmation
# Long when: Alligator bullish alignment (jaw < teeth < lips) AND Bear Power > 0 AND volume > 1.5x 20 EMA
# Short when: Alligator bearish alignment (jaw > teeth > lips) AND Bull Power < 0 AND volume > 1.5x 20 EMA
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-37 trades/year per symbol.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Uses 1d for HTF Elder Ray to avoid counter-trend trades and 6h for Alligator timing.

name = "6h_WilliamsAlligator_1dElderRay_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data ONCE before loop for Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Calculate 6h Williams Alligator (13,8,5 SMAs shifted)
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Jaw (13-period SMA of median price, shifted 8 bars)
    median_6h = (high_6h + low_6h) / 2
    jaw_raw = pd.Series(median_6h).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth (8-period SMA of median price, shifted 5 bars)
    teeth_raw = pd.Series(median_6h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid after shift
    
    # Lips (5-period SMA of median price, shifted 3 bars)
    lips_raw = pd.Series(median_6h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    # Align 6h Alligator to 6h timeframe (no alignment needed as primary TF)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Get 1d data for Elder Ray Power
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray Power (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d  # Bull Power
    bear_power = low_1d - ema_13_1d   # Bear Power
    
    # Align 1d Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish AND Bear Power > 0 AND volume spike
            if (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and 
                bear_power_aligned[i] > 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish AND Bull Power < 0 AND volume spike
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and 
                  bull_power_aligned[i] < 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bear Power <= 0
            if (not (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]) or 
                bear_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bull Power >= 0
            if (not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) or 
                bull_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals