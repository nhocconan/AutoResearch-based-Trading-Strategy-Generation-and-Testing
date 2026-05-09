#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day Chaikin Money Flow (CMF) > 0 for money flow confirmation and 1-week RSI < 30 for oversold conditions.
# In oversold conditions (weekly RSI < 30) with positive money flow (daily CMF > 0), price tends to reverse upward.
# Enters long when both conditions are met, exits when either condition fails.
# Uses weekly RSI for oversold detection and daily CMF for institutional accumulation/distribution.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_CMF_WeeklyRSI_Oversold"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Chaikin Money Flow (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    volume_1d = df_1d['volume']
    
    # Money Flow Multiplier
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)  # Handle division by zero
    
    # Money Flow Volume
    mfv = mfm * volume_1d
    
    # CMF = 20-period sum of MFV / 20-period sum of volume
    cmf = mfv.rolling(window=20, min_periods=20).sum() / volume_1d.rolling(window=20, min_periods=20).sum()
    cmf_values = cmf.values
    cmf_positive = cmf > 0
    cmf_positive_values = cmf_positive.values
    cmf_positive_aligned = align_htf_to_ltf(prices, df_1d, cmf_positive_values)
    
    # Calculate 1-week RSI (14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    delta = close_1w.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_oversold = rsi < 30
    rsi_oversold_values = rsi_oversold.values
    rsi_oversold_aligned = align_htf_to_ltf(prices, df_1w, rsi_oversold_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(cmf_positive_aligned[i]) or
            np.isnan(rsi_oversold_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: daily CMF > 0 (accumulation) + weekly RSI < 30 (oversold)
            if cmf_positive_aligned[i] and rsi_oversold_aligned[i]:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit long: either condition fails
            if not (cmf_positive_aligned[i] and rsi_oversold_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals