#!/usr/bin/env python3
"""
6h_Williams_Alligator_ElderRay
Hypothesis: Williams Alligator identifies trend direction, Elder Ray confirms momentum strength.
Works in bull via Alligator bullish alignment + positive Elder Ray, in bear via bearish alignment + negative Elder Ray.
Target: 15-30 trades/year on 6h timeframe with disciplined entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator (13,8,5 SMAs with 8,5,3 shifts)
    jaw = np.full(n, np.nan)  # 13-period, 8-shift
    teeth = np.full(n, np.nan)  # 8-period, 5-shift
    lips = np.full(n, np.nan)   # 5-period, 3-shift
    
    for i in range(13, n):
        jaw[i] = np.mean(high[i-13:i])  # Using high for jaw per Williams
    for i in range(8, n):
        teeth[i] = np.mean(low[i-8:i])   # Using low for teeth
    for i in range(5, n):
        lips[i] = np.mean(close[i-5:i])  # Using close for lips
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = np.full(n, np.nan)
    k = 2 / (13 + 1)
    for i in range(13, n):
        if i == 13:
            ema13[i] = np.mean(close[0:14])
        else:
            ema13[i] = close[i] * k + ema13[i-1] * (1 - k)
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    k1d = 2 / (34 + 1)
    for i in range(34, len(close_1d)):
        if i == 34:
            ema34_1d[i] = np.mean(close_1d[0:35])
        else:
            ema34_1d[i] = close_1d[i] * k1d + ema34_1d[i-1] * (1 - k1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5)  # Ensure Alligator ready
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Alligator bullish (lips > teeth > jaw) + Bull Power > 0 + price above 1d EMA34
            if (lips[i] > teeth[i] > jaw[i] and 
                bull_power[i] > 0 and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish (jaw > teeth > lips) + Bear Power < 0 + price below 1d EMA34
            elif (jaw[i] > teeth[i] > lips[i] and 
                  bear_power[i] < 0 and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator turns bearish OR Bull Power <= 0
            if not (lips[i] > teeth[i] > jaw[i] and bull_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator turns bullish OR Bear Power >= 0
            if not (jaw[i] > teeth[i] > lips[i] and bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_ElderRay"
timeframe = "6h"
leverage = 1.0