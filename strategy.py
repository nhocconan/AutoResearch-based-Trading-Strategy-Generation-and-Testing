#!/usr/bin/env python3
"""
6h_EMA_Ribbon_Retest_Trend
Hypothesis: Trade EMA ribbon retests in the direction of weekly trend. Uses 8/21/55 EMA ribbon on 6h. When price pulls back to the ribbon (within 0.5% of the 21 EMA) and the weekly trend is up (price > weekly EMA200), go long. When price pulls back to the ribbon and weekly trend is down (price < weekly EMA200), go short. This strategy aims to catch trend continuations with low-risk entries during pullbacks, working in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets. Designed for 6h timeframe to target 15-35 trades/year.
"""

name = "6h_EMA_Ribbon_Retest_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 6h EMA ribbon: 8, 21, 55
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(55, n):
        # Skip if any required data is NaN
        if (np.isnan(ema8[i]) or np.isnan(ema21[i]) or np.isnan(ema55[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Define ribbon area: between 8 and 55 EMA
        ribbon_top = max(ema8[i], ema55[i])
        ribbon_bottom = min(ema8[i], ema55[i])
        
        # Price near 21 EMA (within 0.5%)
        near_ema21 = abs(close[i] - ema21[i]) / ema21[i] < 0.005
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        if position == 0:
            # LONG: Price near 21 EMA from below AND weekly uptrend
            if near_ema21 and weekly_uptrend and low[i] <= ribbon_top and high[i] >= ribbon_bottom:
                signals[i] = 0.25
                position = 1
            # SHORT: Price near 21 EMA from above AND weekly downtrend
            elif near_ema21 and weekly_downtrend and low[i] <= ribbon_top and high[i] >= ribbon_bottom:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 8 EMA or weekly trend turns down
            if close[i] < ema8[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 55 EMA or weekly trend turns up
            if close[i] > ema55[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals