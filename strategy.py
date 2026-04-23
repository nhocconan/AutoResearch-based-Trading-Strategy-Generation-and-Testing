#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d Elder Ray trend filter and volume confirmation
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend via aligned SMAs
- 1d Elder Ray (Bull Power/Bear Power) confirms higher timeframe trend strength
- Volume spike (> 2.0x 20-period average) filters false signals
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1d Elder Ray trend
- Alligator provides trend direction, Elder Ray adds momentum confirmation
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
    
    # Calculate 12h Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1d Elder Ray for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20)  # for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish alignment + 1d Bull Power positive + volume
            alligator_bullish = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
            elder_bullish = bull_power_aligned[i] > 0
            volume_confirmed = volume[i] > 2.0 * vol_ma[i]
            
            # Short conditions: Alligator bearish alignment + 1d Bear Power negative + volume
            alligator_bearish = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
            elder_bearish = bear_power_aligned[i] < 0
            
            if alligator_bullish and elder_bullish and volume_confirmed:
                signals[i] = 0.25
                position = 1
            elif alligator_bearish and elder_bearish and volume_confirmed:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator sleeping (intertwined) or Elder Ray divergence
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator loses bullish alignment or 1d Bull Power turns negative
                alligator_bullish = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
                elder_bullish = bull_power_aligned[i] > 0
                if not (alligator_bullish and elder_bullish):
                    exit_signal = True
            elif position == -1:
                # Exit short: Alligator loses bearish alignment or 1d Bear Power turns positive
                alligator_bearish = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
                elder_bearish = bear_power_aligned[i] < 0
                if not (alligator_bearish and elder_bearish):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dElderRay_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0