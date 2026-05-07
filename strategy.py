#!/usr/bin/env python3
"""
6h_Chaikin_Money_Flow_Riverbank
Hypothesis: Chaikin Money Flow (CMF) combined with Riverbank EMA crossover identifies accumulation/distribution phases with trend confirmation. Works in bull/bear markets by detecting institutional flow direction while avoiding whipsaws. Target: 12-30 trades/year.
"""
name = "6h_Chaikin_Money_Flow_Riverbank"
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
    
    # Get 12h data for CMF calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    # Calculate Chaikin Money Flow (21-period) on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Money Flow Multiplier
    mfm = np.where((high_12h - low_12h) != 0, 
                   ((close_12h - low_12h) - (high_12h - close_12h)) / (high_12h - low_12h), 
                   0)
    # Money Flow Volume
    mfv = mfm * volume_12h
    # CMF: 21-period sum of MFV / 21-period sum of volume
    mfv_sum = pd.Series(mfv).rolling(window=21, min_periods=21).sum().values
    vol_sum = pd.Series(volume_12h).rolling(window=21, min_periods=21).sum().values
    cmf_12h = np.where(vol_sum != 0, mfv_sum / vol_sum, 0)
    
    # Align CMF to 6h timeframe
    cmf_12h_aligned = align_htf_to_ltf(prices, df_12h, cmf_12h)
    
    # Riverbank EMA crossover on 6h: EMA(9) and EMA(20)
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(cmf_12h_aligned[i]) or np.isnan(ema_9[i]) or 
            np.isnan(ema_20[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CMF > 0.05 (accumulation) + EMA9 > EMA20 (uptrend) + volume
            if cmf_12h_aligned[i] > 0.05 and ema_9[i] > ema_20[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.05 (distribution) + EMA9 < EMA20 (downtrend) + volume
            elif cmf_12h_aligned[i] < -0.05 and ema_9[i] < ema_20[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: CMF crosses back to neutral zone or EMA cross reverses
            if position == 1:
                if cmf_12h_aligned[i] < 0 or ema_9[i] < ema_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if cmf_12h_aligned[i] > 0 or ema_9[i] > ema_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals