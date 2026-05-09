#!/usr/bin/env python3
name = "6H_Daily_Alligator_Trend_Momentum"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Alligator and momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate daily Alligator lines (Williams Alligator)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Jaw (Blue Line): 13-period SMMA, 8 bars ahead
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (Red Line): 8-period SMMA, 5 bars ahead
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (Green Line): 5-period SMMA, 3 bars ahead
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Momentum: Daily ROC(5)
    roc5_1d = np.diff(np.concatenate([[np.nan], close_1d])) / np.concatenate([[np.nan], close_1d[:-1]]) * 100
    roc5_1d = pd.Series(roc5_1d).rolling(window=5, min_periods=5).mean().values
    roc5_aligned = align_htf_to_ltf(prices, df_1d, roc5_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(roc5_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Lips above Teeth above Jaw (bullish alignment) + positive momentum
            if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and roc5_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips below Teeth below Jaw (bearish alignment) + negative momentum
            elif lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and roc5_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Lips cross below Teeth (trend weakening) or momentum turns negative
            if lips_aligned[i] < teeth_aligned[i] or roc5_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Lips cross above Teeth (trend weakening) or momentum turns positive
            if lips_aligned[i] > teeth_aligned[i] or roc5_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals