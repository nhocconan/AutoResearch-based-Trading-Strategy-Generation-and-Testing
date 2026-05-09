#!/usr/bin/env python3
# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume spike
# Long when price > Alligator teeth, green mouth alignment, EMA50 uptrend, volume > 2x average
# Short when price < Alligator teeth, red mouth alignment, EMA50 downtrend, volume > 2x average
# Exit when price crosses Alligator jaws or trend reverses
# Uses Alligator (SMAs with specific offsets) for trend identification and market structure
# Designed to capture trends with clear entry/exit rules and controlled frequency
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "4h_Williams_Alligator_EMA50_Volume"
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
    
    # Calculate 1d Williams Alligator (Jaw, Teeth, Lips)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Alligator components: SMAs with specific periods and shifts
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values  # Jaw: SMA(13) shifted 8
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values   # Teeth: SMA(8) shifted 5
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values    # Lips: SMA(5) shifted 3
    
    # Align Alligator components to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Alligator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > teeth, green mouth (lips > teeth > jaw), EMA50 up, volume spike
            if (close[i] > teeth_aligned[i] and 
                lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < teeth, red mouth (lips < teeth < jaw), EMA50 down, volume spike
            elif (close[i] < teeth_aligned[i] and 
                  lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below jaws or trend reverses
            if (close[i] < jaw_aligned[i]) or (ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above jaws or trend reverses
            if (close[i] > jaw_aligned[i]) or (ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals