#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray + Volume Confirmation
# Uses Williams Alligator (3 SMAs) from 1d to define trend direction (jaws/teeth/lips alignment)
# Elder Ray (bull/bear power) from 6h to measure trend strength and momentum
# Volume spike (>1.5x 20-bar average) confirms institutional participation
# Designed to work in both bull and bear markets by following the 1d trend via Alligator
# Entry only when Alligator is aligned (bullish/bearish) AND Elder Ray confirms strength
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25
# Williams Alligator reduces whipsaw in ranging markets, Elder Ray filters weak moves

name = "6h_WilliamsAlligator_ElderRay_Volume_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator from 1d: Jaw(13,8), Teeth(8,5), Lips(5,3)
    # Smoothed SMAs with offset as per Williams
    close_1d_series = pd.Series(close_1d)
    jaw = close_1d_series.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = close_1d_series.rolling(window=8, min_periods=8).mean().shift(5).values
    lips = close_1d_series.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator components to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray from 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: >1.5x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator alignment: Lips > Teeth > Jaw AND bullish power positive
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) and (bull_power[i] > 0) and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Bearish Alligator alignment: Jaw > Teeth > Lips AND bearish power negative
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) and (bear_power[i] < 0) and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR bull power turns negative
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or (bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR bear power turns positive
            if not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) or (bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals