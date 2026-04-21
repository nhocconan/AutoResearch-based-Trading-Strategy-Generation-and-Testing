#!/usr/bin/env python3
"""
4h_ChaikinMoneyFlow_1dTrend_Filter_V1
Hypothesis: Chaikin Money Flow (CMF) on 4h detects institutional buying/selling pressure. 
Only take long when CMF > 0.05 and price > 1d EMA50 (bullish trend); short when CMF < -0.05 and price < 1d EMA50 (bearish trend).
Uses 1d EMA50 as trend filter to avoid counter-trend trades. CMF threshold reduces whipsaw.
Works in bull/bear by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 1d data once for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] - ema_50[i-1]) * multiplier + ema_50[i-1]
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Chaikin Money Flow (CMF) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Money Flow Multiplier
    mfm = np.zeros_like(close)
    for i in range(n):
        if high[i] == low[i]:
            mfm[i] = 0.0
        else:
            mfm[i] = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # 20-period CMF
    cmf = np.zeros(n)
    for i in range(n):
        if i < 19:
            cmf[i] = 0.0
        else:
            sum_mfv = np.sum(mfv[i-19:i+1])
            sum_volume = np.sum(volume[i-19:i+1])
            if sum_volume > 0:
                cmf[i] = sum_mfv / sum_volume
            else:
                cmf[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if EMA50 not available
        if np.isnan(ema_50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50 = ema_50_aligned[i]
        cmf_val = cmf[i]
        
        if position == 0:
            # Long: CMF > 0.05 and price above 1d EMA50 (bullish trend)
            if cmf_val > 0.05 and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.05 and price below 1d EMA50 (bearish trend)
            elif cmf_val < -0.05 and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CMF turns negative or price drops below EMA50
            if cmf_val < 0 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CMF turns positive or price rises above EMA50
            if cmf_val > 0 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ChaikinMoneyFlow_1dTrend_Filter_V1"
timeframe = "4h"
leverage = 1.0