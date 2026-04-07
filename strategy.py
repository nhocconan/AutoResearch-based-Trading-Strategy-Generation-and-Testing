#!/usr/bin/env python3
"""
6H Elder Ray Momentum with 1w Trend Filter
Long when Bull Power > 0 AND Bear Power < 0 AND 1w EMA trend up
Short when Bear Power < 0 AND Bull Power > 0 AND 1w EMA trend down
Exit when Bull Power * Bear Power > 0 (both same sign)
Elder Ray uses EMA13: Bull = High - EMA13, Bear = Low - EMA13
Works in bull/bear by requiring trend alignment from 1w timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_momentum_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === EMA 13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Elder Ray Components ===
    bull_power = high - ema13  # Strength of bulls: ability to push above EMA
    bear_power = low - ema13   # Strength of bears: ability to push below EMA
    
    # === 1w trend filter (EMA 21) ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(ema_1w_aligned[i-1])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: when bull and bear power same sign (both + or both -)
            if bull_power[i] * bear_power[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: when bull and bear power same sign
            if bull_power[i] * bear_power[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Entry: Bullish when bulls strong (+), bears weak (-) AND 1w uptrend
            if bull_power[i] > 0 and bear_power[i] < 0 and ema_1w_aligned[i] > ema_1w_aligned[i-1]:
                position = 1
                signals[i] = 0.25
            # Entry: Bearish when bears strong (-), bulls weak (+) AND 1w downtrend
            elif bear_power[i] < 0 and bull_power[i] > 0 and ema_1w_aligned[i] < ema_1w_aligned[i-1]:
                position = -1
                signals[i] = -0.25
    
    return signals