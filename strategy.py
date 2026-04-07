#!/usr/bin/env python3
"""
6h_elder_ray_1d_power_v1
Hypothesis: Elder Ray index (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6f with 1d EMA50 trend filter.
In bull markets (price > EMA50), buy when Bull Power turns positive after being negative.
In bear markets (price < EMA50), sell when Bear Power turns negative after being positive.
Uses zero-crossings of power indicators to capture momentum shifts while filtering by higher timeframe trend.
Works in both bull and bear by aligning with 1d trend direction.
Target: 12-37 trades/year on 6f with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_1d_power_v1"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (on 6f data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if data not available
        if np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter
        above_ema50 = close[i] > ema50_1d_aligned[i]
        below_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Bear Power turns negative (momentum fading)
            if bear_power[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive (momentum fading)
            if bull_power[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # In uptrend: buy when Bull Power turns positive from negative
            if above_ema50 and bull_power[i] > 0 and bull_power[i-1] <= 0:
                position = 1
                signals[i] = 0.25
            # In downtrend: sell when Bear Power turns negative from positive
            elif below_ema50 and bear_power[i] < 0 and bear_power[i-1] >= 0:
                position = -1
                signals[i] = -0.25
    
    return signals