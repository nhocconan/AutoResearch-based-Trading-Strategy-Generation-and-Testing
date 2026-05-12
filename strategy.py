#!/usr/bin/env python3
# 1d_WeeklyPivot_200EMA_TrendFilter
# Hypothesis: On 1d timeframe, enter long when price closes above weekly pivot with close > 200 EMA.
# Enter short when price closes below weekly pivot with close < 200 EMA.
# Exit when price crosses 200 EMA (trend reversal).
# Uses 200 EMA for trend filter to reduce whipsaw and improve performance in both bull and bear markets.
# Targets 10-20 trades/year for low fee drag.

name = "1d_WeeklyPivot_200EMA_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 3:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Calculate 200 EMA
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly pivot to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure EMA200 is stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(pivot_aligned[i]) or np.isnan(ema200[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        pivot_val = pivot_aligned[i]
        ema200_val = ema200[i]
        
        if position == 0:
            # LONG: Price closes above weekly pivot with close > 200 EMA
            if close[i] > pivot_val and close[i] > ema200_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below weekly pivot with close < 200 EMA
            elif close[i] < pivot_val and close[i] < ema200_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 200 EMA (trend reversal)
            if close[i] < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 200 EMA (trend reversal)
            if close[i] > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals