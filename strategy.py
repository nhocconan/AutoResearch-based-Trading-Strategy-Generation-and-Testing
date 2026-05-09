#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with Elder Ray power trend filter and volume confirmation
# Long when price > Alligator teeth, Bull Power > 0, and volume > 1.5x average
# Short when price < Alligator teeth, Bear Power < 0, and volume > 1.5x average
# Exit when price crosses Alligator jaw or power signals reverse
# Uses Williams Alligator (SMAs) for trend, Elder Ray for bull/bear power, volume for conviction
# Designed to capture trends in both bull and bear markets with controlled frequency
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_Alligator_ElderRay_PowerTrend"
timeframe = "12h"
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
    
    # Calculate 1w Williams Alligator (Jaw, Teeth, Lips)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # Using SMA as approximation for SMMA (simple moving average)
    close_1w = df_1w['close'].values
    jaw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Calculate 1d Elder Ray (Bull Power, Bear Power)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(teeth_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > teeth, bull power > 0, volume spike
            if (close[i] > teeth_aligned[i] and 
                bull_power_aligned[i] > 0 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < teeth, bear power < 0, volume spike
            elif (close[i] < teeth_aligned[i] and 
                  bear_power_aligned[i] < 0 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses jaw or bull power turns negative
            if (close[i] < jaw_aligned[i]) or (bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses jaw or bear power turns positive
            if (close[i] > jaw_aligned[i]) or (bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals