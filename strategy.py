#!/usr/bin/env python3
"""
6h_Aroon_Trend_Strength
Hypothesis: Aroon indicator identifies trend strength and direction (Aroon Up/Down). 
Long when Aroon Up > 70 and Aroon Down < 30, indicating strong uptrend.
Short when Aroon Down > 70 and Aroon Up < 30, indicating strong downtrend.
Uses 12h EMA50 as trend filter to avoid counter-trend trades. Works in trending markets by capturing sustained moves.
Target: 15-35 trades/year per symbol.
"""

name = "6h_Aroon_Trend_Strength"
timeframe = "6h"
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
    
    # Aroon indicator (period 25)
    period = 25
    aroon_up = np.zeros(n)
    aroon_down = np.zeros(n)
    
    for i in range(period - 1, n):
        # Periods since highest high
        highest_high_idx = i - np.argmax(high[i - period + 1:i + 1])
        periods_since_high = i - highest_high_idx
        aroon_up[i] = ((period - periods_since_high) / period) * 100
        
        # Periods since lowest low
        lowest_low_idx = i - np.argmin(low[i - period + 1:i + 1])
        periods_since_low = i - lowest_low_idx
        aroon_down[i] = ((period - periods_since_low) / period) * 100
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = df_12h['close'].values > ema_50_12h
    downtrend_12h = df_12h['close'].values < ema_50_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(period - 1, n):
        up = aroon_up[i]
        down = aroon_down[i]
        
        uptrend_htf = uptrend_12h_aligned[i]
        downtrend_htf = downtrend_12h_aligned[i]
        
        if position == 0:
            # LONG: Aroon Up > 70, Aroon Down < 30, 12h uptrend
            if up > 70 and down < 30 and uptrend_htf:
                signals[i] = 0.25
                position = 1
            # SHORT: Aroon Down > 70, Aroon Up < 30, 12h downtrend
            elif down > 70 and up < 30 and downtrend_htf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Aroon Down > 50 or Aroon Up < 50 (trend weakening)
            if down > 50 or up < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Aroon Up > 50 or Aroon Down < 50 (trend weakening)
            if up > 50 or down < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals