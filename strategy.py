#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume spike confirmation
- Uses Williams Alligator (JAW=13, TEETH=8, LIPS=5) from 12h timeframe for trend identification
- 1d EMA(34) defines higher timeframe trend (only long when price > EMA, short when price < EMA)
- Volume confirmation (> 1.8x 20-period average) filters low-momentum signals
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend
- Williams Alligator provides smooth trend identification with built-in filters
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
    
    # Calculate 12h Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_12h = (df_12h['high'] + df_12h['low']) / 2
    jaw = pd.Series(median_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(median_12h).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(median_12h).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 34, 20)  # Alligator jaw, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator conditions: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        bullish_alligator = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        bearish_alligator = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long conditions: bullish Alligator, uptrend, volume spike
            long_signal = (bullish_alligator and 
                          uptrend and
                          volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions: bearish Alligator, downtrend, volume spike
            short_signal = (bearish_alligator and 
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
                # Exit long: bearish Alligator or trend turns down
                if (bearish_alligator or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: bullish Alligator or trend turns up
                if (bullish_alligator or 
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