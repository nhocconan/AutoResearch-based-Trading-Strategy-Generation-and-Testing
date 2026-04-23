#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation
- Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) from 1d timeframe to identify trend direction
- 12h EMA(34) acts as additional trend confirmation (only long when price > EMA, short when price < EMA)
- Volume confirmation (> 1.8x 30-period average) filters low-momentum signals
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the Alligator/EMA trend alignment
- Volume spike requirement reduces false signals during low volatility
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5) - all shifted forward
    jaw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 12h EMA(34) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Volume confirmation: > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 34, 30)  # Alligator, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Alligator alignment (all lines aligned in same direction)
        alligator_long = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        alligator_short = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long conditions: Alligator aligned up, uptrend, volume spike
            long_signal = (alligator_long and 
                          uptrend and
                          volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions: Alligator aligned down, downtrend, volume spike
            short_signal = (alligator_short and 
                           downtrend and
                           volume[i] > 1.8 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator reversal or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator turns down OR trend turns down
                if (not alligator_long or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Alligator turns up OR trend turns up
                if (not alligator_short or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0