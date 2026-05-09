#!/usr/bin/env python3
# 12h_Chaikin_Money_Flow_1wTrend_Ref
# Hypothesis: Chaikin Money Flow (CMF) on 12h with 1-week trend filter and volume confirmation.
# Works in bull/bear: 1w trend filter ensures trades align with higher-timeframe momentum,
# CMF detects institutional accumulation/distribution, volume filter confirms strength.
# Uses 12h timeframe to balance signal frequency and cost efficiency.

name = "12h_Chaikin_Money_Flow_1wTrend_Ref"
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
    
    # Calculate 1-week EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[0:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (ema_200_1w[i-1] * 199 + close_1w[i]) / 200
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Chaikin Money Flow (CMF) on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
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
    
    # 20-period CMF
    cmf = np.full_like(close_12h, np.nan)
    if len(mfv) >= 20:
        # Initialize first valid value
        cmf[19] = np.sum(mfv[0:20]) / np.sum(volume_12h[0:20])
        # Rolling calculation
        for i in range(20, len(mfv)):
            cmf[i] = (np.sum(mfv[i-19:i+1]) / np.sum(volume_12h[i-19:i+1]))
    
    cmf_aligned = align_htf_to_ltf(prices, df_12h, cmf)
    
    # Volume confirmation: current volume / 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure CMF and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(cmf_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: CMF > 0.1 (accumulation) AND uptrend (price > EMA200) AND volume spike
            if (cmf_aligned[i] > 0.1 and 
                close[i] > ema_200_1w_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: CMF < -0.1 (distribution) AND downtrend (price < EMA200) AND volume spike
            elif (cmf_aligned[i] < -0.1 and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: CMF < 0 (loss of accumulation) OR trend reversal (price < EMA200)
            if cmf_aligned[i] < 0 or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: CMF > 0 (loss of distribution) OR trend reversal (price > EMA200)
            if cmf_aligned[i] > 0 or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals