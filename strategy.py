#!/usr/bin/env python3
# 6h_1d_elder_ray_ema200_trend
# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with EMA200 filter on daily trend.
# Elder Ray measures bull/bear power relative to EMA13. Combined with daily EMA200 trend filter,
# it captures momentum in trending markets while avoiding counter-trend trades in chop.
# Works in bull/bear by only taking trades aligned with higher timeframe trend.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

name = "6h_1d_elder_ray_ema200_trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 6-hour EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if EMA200 not ready
        if np.isnan(ema200_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine daily trend: above EMA200 = uptrend, below = downtrend
        uptrend = close[i] > ema200_1d_aligned[i]
        downtrend = close[i] < ema200_1d_aligned[i]
        
        # Long entry: uptrend + bull power positive and increasing
        if uptrend and bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: downtrend + bear power negative and decreasing (more negative)
        elif downtrend and bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal or power divergence
        elif position == 1 and (not uptrend or bull_power[i] < 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not downtrend or bear_power[i] > 0):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals