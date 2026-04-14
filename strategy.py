# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h Ehlers Fisher Transform on 1d closes, filtered by 1w Supertrend direction.
Long when Fisher crosses above -1.5 and price > 1w Supertrend; short when Fisher crosses below +1.5 and price < 1w Supertrend.
Exit on opposite Fisher cross. Targets 15-25 trades/year per symbol (60-100 total over 4 years) with low frequency to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ehlers_fisher_transform(price, length):
    """Ehlers Fisher Transform: normalizes price to [-1, 1] range, applies 0.5*ln((1+x)/(1-x)), then smooths."""
    n = len(price)
    if n < length:
        return np.full(n, np.nan)
    
    # Highest high and lowest low over lookback
    highest = pd.Series(price).rolling(window=length, min_periods=length).max().values
    lowest = pd.Series(price).rolling(window=length, min_periods=length).min().values
    
    # Avoid division by zero
    diff = highest - lowest
    diff[diff == 0] = 1e-10
    
    # Normalize price to [-1, 1]
    value1 = 2 * ((price - lowest) / diff) - 1
    
    # Clamp to avoid domain error in arctanh
    value1 = np.clip(value1, -0.999, 0.999)
    
    # Fisher transform: 0.5 * ln((1+x)/(1-x))
    fish = 0.5 * np.log((1 + value1) / (1 - value1))
    
    # Smooth with exponential moving average
    fish_smoothed = pd.Series(fish).ewm(alpha=0.5, adjust=False).fillna(0).values
    
    return fish_smoothed

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Average True Range
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, fillna=0).values
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.full(n, np.nan)
    final_lb = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    trend = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(period, n):
        if i == period:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            final_ub[i] = basic_ub[i] if (basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]) else final_ub[i-1]
            final_lb[i] = basic_lb[i] if (basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]) else final_lb[i-1]
        
        if i == period:
            supertrend[i] = final_ub[i]
            trend[i] = -1
        else:
            if supertrend[i-1] == final_ub[i-1]:
                supertrend[i] = final_lb[i] if close[i] <= final_lb[i] else final_ub[i]
                trend[i] = -1 if supertrend[i] == final_ub[i] else 1
            else:
                supertrend[i] = final_ub[i] if close[i] >= final_ub[i] else final_lb[i]
                trend[i] = 1 if supertrend[i] == final_ub[i] else -1
    
    return supertrend, trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Fisher Transform
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Fisher Transform on 1d close
    fish = ehlers_fisher_transform(df_1d['close'].values, 10)
    fish_aligned = align_htf_to_ltf(prices, df_1d, fish)
    
    # Get 1w data for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Supertrend on 1w
    supertrend_1w, trend_1w = calculate_supertrend(
        df_1w['high'].values,
        df_1w['low'].values,
        df_1w['close'].values,
        period=10,
        multiplier=3.0
    )
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(30, n):
        # Skip if any values are NaN
        if (np.isnan(fish_aligned[i]) or 
            np.isnan(supertrend_aligned[i])):
            continue
        
        fish_val = fish_aligned[i]
        fish_prev = fish_aligned[i-1] if i > 0 else fish_val
        price = close[i]
        st = supertrend_aligned[i]
        
        if position == 0:  # No position - look for entries
            # Long: Fisher crosses above -1.5 and price above Supertrend
            if fish_prev <= -1.5 and fish_val > -1.5 and price > st:
                position = 1
                signals[i] = position_size
            # Short: Fisher crosses below +1.5 and price below Supertrend
            elif fish_prev >= 1.5 and fish_val < 1.5 and price < st:
                position = -1
                signals[i] = -position_size
        elif position == 1:  # Long position - exit when Fisher crosses below +1.5
            if fish_prev >= 1.5 and fish_val < 1.5:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when Fisher crosses above -1.5
            if fish_prev <= -1.5 and fish_val > -1.5:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_Fisher_1d_Supertrend_1w"
timeframe = "6h"
leverage = 1.0