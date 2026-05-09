#!/usr/bin/env python3
# Hypothesis: 4h Williams Alligator with 12h Elder Ray and volume confirmation
# Long when Alligator bullish (JAW<TEETH<LIPS), Elder Ray bullish, and volume > 1.5x average
# Short when Alligator bearish (JAW>TEETH>LIPS), Elder Ray bearish, and volume > 1.5x average
# Exit when Alligator reverses (JAW crosses TEETH) or volume drops below average
# Uses Alligator for trend direction, Elder Ray for bull/bear power, volume for conviction
# Designed to capture trends in both bull and bear markets with controlled frequency
# Target: 75-150 total trades over 4 years (19-38/year) with size 0.25

name = "4h_Williams_Alligator_ElderRay_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams Alligator (13,8,5 SMAs with 8,5,3 offsets)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Jaw: 13-period SMA, 8 bars ahead
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, 5 bars ahead
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, 3 bars ahead
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator components to 4h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Calculate 12h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema13_12h = pd.Series(df_12h['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_12h = df_12h['high'].values - ema13_12h
    bear_power_12h = df_12h['low'].values - ema13_12h
    
    # Align Elder Ray components to 4h timeframe
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or np.isnan(lips_12h_aligned[i]) or
            np.isnan(bull_power_12h_aligned[i]) or np.isnan(bear_power_12h_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Alligator bullish (JAW<TEETH<LIPS), Elder Ray bullish, volume spike
            if (jaw_12h_aligned[i] < teeth_12h_aligned[i] < lips_12h_aligned[i] and
                bull_power_12h_aligned[i] > 0 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator bearish (JAW>TEETH>LIPS), Elder Ray bearish, volume spike
            elif (jaw_12h_aligned[i] > teeth_12h_aligned[i] > lips_12h_aligned[i] and
                  bear_power_12h_aligned[i] < 0 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator reverses (JAW crosses above TEETH) or volume drops
            if (jaw_12h_aligned[i] >= teeth_12h_aligned[i]) or (not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator reverses (JAW crosses below TEETH) or volume drops
            if (jaw_12h_aligned[i] <= teeth_12h_aligned[i]) or (not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals