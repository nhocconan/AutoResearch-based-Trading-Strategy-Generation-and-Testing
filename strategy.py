#!/usr/bin/env python3
"""
6h_Chaikin_Money_Flow_Trend_Filter
Hypothesis: Chaikin Money Flow (CMF) measures institutional money flow strength. Long when CMF > 0.15 with 12h EMA50 uptrend, short when CMF < -0.15 with 12h EMA50 downtrend. CMF filters false breakouts by requiring volume-price alignment, working in both bull (accumulation) and bear (distribution) markets. Target: 20-40 trades/year per symbol.
"""

name = "6h_Chaikin_Money_Flow_Trend_Filter"
timeframe = "6h"
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
    
    # Chaikin Money Flow (20-period)
    # CMF = Sum((Close - Low - (High - Close)) / (High - Low) * Volume) / Sum(Volume)
    # Simplified: ((Close - Low) - (High - Close)) / (High - Low) = (2*Close - High - Low)/(High - Low)
    mfm = ((2 * close - high - low) / (high - low))
    mfm = np.where((high - low) == 0, 0, mfm)  # avoid division by zero
    mfv = mfm * volume
    cmf = pd.Series(mfv).rolling(window=20, min_periods=20).sum() / pd.Series(volume).rolling(window=20, min_periods=20).sum()
    cmf = cmf.values
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(cmf[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: CMF > 0.15 (accumulation) + 12h EMA50 uptrend
            if (cmf[i] > 0.15 and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: CMF < -0.15 (distribution) + 12h EMA50 downtrend
            elif (cmf[i] < -0.15 and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF < 0 OR closes below 12h EMA50
            if (cmf[i] < 0) or \
               (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF > 0 OR closes above 12h EMA50
            if (cmf[i] > 0) or \
               (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals