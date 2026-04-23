#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA trend filter and volume confirmation
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) defines market structure and trend
- 1w EMA(34) confirms higher timeframe trend direction
- Volume confirmation (> 1.3x 20-period average) filters low-momentum signals
- Designed for 1d timeframe targeting 15-30 trades/year (60-120 over 4 years)
- Alligator provides natural trend/filter: long when Lips > Teeth > Jaw, short when reversed
- Works in both bull and bear markets by trading with 1w trend and Alligator alignment
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
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator levels to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1w EMA(34) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 8, 5, 34, 20)  # Alligator, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Alligator alignment (trend direction)
        # Long alignment: Lips > Teeth > Jaw (bullish alignment)
        # Short alignment: Lips < Teeth < Jaw (bearish alignment)
        bullish_align = (lips_aligned[i] > teeth_aligned[i] and 
                        teeth_aligned[i] > jaw_aligned[i])
        bearish_align = (lips_aligned[i] < teeth_aligned[i] and 
                        teeth_aligned[i] < jaw_aligned[i])
        
        # Trend filter: price vs 1w EMA
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long conditions: bullish Alligator alignment, uptrend, volume spike
            long_signal = (bullish_align and 
                          uptrend and
                          volume[i] > 1.3 * vol_ma[i])
            
            # Short conditions: bearish Alligator alignment, downtrend, volume spike
            short_signal = (bearish_align and 
                           downtrend and
                           volume[i] > 1.3 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Alligator alignment or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish Alligator alignment or trend turns down
                if (bearish_align or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: bullish Alligator alignment or trend turns up
                if (bullish_align or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA34_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0