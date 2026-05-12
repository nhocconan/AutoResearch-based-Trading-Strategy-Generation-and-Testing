#!/usr/bin/env python3
# 6H_ELDER_RAY_POWER_1D_TREND_FILTER
# Hypothesis: On 6h timeframe, use Elder Ray Bull/Bear Power from daily timeframe to detect institutional pressure.
# Enter long when Bull Power > 0 and Bear Power < 0 with 6h price above EMA20 (bullish alignment).
# Enter short when Bear Power > 0 and Bull Power < 0 with 6h price below EMA20 (bearish alignment).
# Exit when power signals weaken or reverse.
# Uses 1d trend filter (EMA50) to avoid counter-trend trades.
# Designed to work in both bull and bear markets via trend filter and power balance.
# Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

name = "6H_ELDER_RAY_POWER_1D_TREND_FILTER"
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
    
    # Elder Ray components from daily timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray (standard setting)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_1d - ema13_1d
    bear_power = ema13_1d - low_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h EMA20 for entry filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power positive, Bear Power negative, price above EMA20, and daily uptrend
            if (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and 
                close[i] > ema20[i] and close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power positive, Bull Power negative, price below EMA20, and daily downtrend
            elif (bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0 and 
                  close[i] < ema20[i] and close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Power weakens or reverses
            if bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Power weakens or reverses
            if bear_power_aligned[i] <= 0 or bull_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals