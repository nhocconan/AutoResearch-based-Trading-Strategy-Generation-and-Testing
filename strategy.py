#!/usr/bin/env python3
"""
12h_Supertrend_1wTrend
Hypothesis: Use 12h Supertrend for entry timing with 1-week Supertrend trend filter.
Go long when 12h Supertrend turns bullish and weekly trend is bullish, short when 12h Supertrend turns bearish and weekly trend is bearish.
Designed for 12h timeframe to capture medium-term trends with very low trade frequency (~10-20/year).
"""

name = "12h_Supertrend_1wTrend"
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
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 12h Supertrend (ATR=10, multiplier=3)
    atr_period = 10
    multiplier = 3
    
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR
    atr = np.zeros_like(close)
    atr[atr_period] = np.mean(tr[1:atr_period+1])
    for i in range(atr_period+1, len(close)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    supertrend[atr_period] = upper_band[atr_period]
    direction[atr_period] = 1
    
    for i in range(atr_period+1, len(close)):
        if close[i] <= upper_band[i-1]:
            upper_band[i] = upper_band[i-1]
        else:
            upper_band[i] = upper_band[i]
        
        if close[i] >= lower_band[i-1]:
            lower_band[i] = lower_band[i-1]
        else:
            lower_band[i] = lower_band[i]
        
        if direction[i-1] == 1 and close[i] < lower_band[i]:
            direction[i] = -1
            supertrend[i] = upper_band[i]
        elif direction[i-1] == -1 and close[i] > upper_band[i]:
            direction[i] = 1
            supertrend[i] = lower_band[i]
        elif direction[i-1] == 1 and close[i] > upper_band[i]:
            direction[i] = 1
            supertrend[i] = lower_band[i]
        elif direction[i-1] == -1 and close[i] < lower_band[i]:
            direction[i] = -1
            supertrend[i] = upper_band[i]
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
    
    # Calculate weekly Supertrend for trend filter
    atr_period_1w = 10
    multiplier_1w = 3
    
    tr1w = high_1w[1:] - low_1w[1:]
    tr2w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3w = np.abs(low_1w[1:] - close_1w[:-1])
    trw = np.concatenate([[np.inf], np.maximum(tr1w, np.maximum(tr2w, tr3w))])
    
    atrw = np.zeros_like(close_1w)
    atrw[atr_period_1w] = np.mean(trw[1:atr_period_1w+1])
    for i in range(atr_period_1w+1, len(close_1w)):
        atrw[i] = (atrw[i-1] * (atr_period_1w-1) + trw[i]) / atr_period_1w
    
    hl2w = (high_1w + low_1w) / 2
    upper_bandw = hl2w + multiplier_1w * atrw
    lower_bandw = hl2w - multiplier_1w * atrw
    
    supertrendw = np.zeros_like(close_1w)
    directionw = np.ones_like(close_1w)
    
    supertrendw[atr_period_1w] = upper_bandw[atr_period_1w]
    directionw[atr_period_1w] = 1
    
    for i in range(atr_period_1w+1, len(close_1w)):
        if close_1w[i] <= upper_bandw[i-1]:
            upper_bandw[i] = upper_bandw[i-1]
        else:
            upper_bandw[i] = upper_bandw[i]
        
        if close_1w[i] >= lower_bandw[i-1]:
            lower_bandw[i] = lower_bandw[i-1]
        else:
            lower_bandw[i] = lower_bandw[i]
        
        if directionw[i-1] == 1 and close_1w[i] < lower_bandw[i]:
            directionw[i] = -1
            supertrendw[i] = upper_bandw[i]
        elif directionw[i-1] == -1 and close_1w[i] > upper_bandw[i]:
            directionw[i] = 1
            supertrendw[i] = lower_bandw[i]
        elif directionw[i-1] == 1 and close_1w[i] > upper_bandw[i]:
            directionw[i] = 1
            supertrendw[i] = lower_bandw[i]
        elif directionw[i-1] == -1 and close_1w[i] < lower_bandw[i]:
            directionw[i] = -1
            supertrendw[i] = upper_bandw[i]
        else:
            directionw[i] = directionw[i-1]
            if directionw[i] == 1:
                supertrendw[i] = lower_bandw[i]
            else:
                supertrendw[i] = upper_bandw[i]
    
    # Align weekly Supertrend direction to 12h timeframe
    directionw_aligned = align_htf_to_ltf(prices, df_1w, directionw.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend[i]) or np.isnan(directionw_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend: 1 for uptrend, -1 for downtrend
        weekly_trend = directionw_aligned[i]
        
        if position == 0:
            # LONG: 12h Supertrend turns bullish AND weekly trend is bullish
            if direction[i] == 1 and direction[i-1] == -1 and weekly_trend == 1:
                signals[i] = 0.25
                position = 1
            # SHORT: 12h Supertrend turns bearish AND weekly trend is bearish
            elif direction[i] == -1 and direction[i-1] == 1 and weekly_trend == -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 12h Supertrend turns bearish
            if direction[i] == -1 and direction[i-1] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 12h Supertrend turns bullish
            if direction[i] == 1 and direction[i-1] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals