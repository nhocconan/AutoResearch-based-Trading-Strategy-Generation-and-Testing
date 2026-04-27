#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_Filter
Hypothesis: On 6h timeframe, Ichimoku cloud twist (Senkou Span A/B cross) combined with 1d trend filter (price > EMA50 for longs, < EMA50 for shorts) captures strong trend continuations while avoiding false signals in ranging markets. The cloud twist indicates momentum shift, and 1d EMA50 ensures alignment with higher timeframe trend. This strategy targets 50-150 trades over 4 years (12-37/year) with discrete position sizing (0.25) to manage fee drag and drawdown. Works in both bull and bear markets by following the 1d trend direction only.
"""

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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Cloud twist: Senkou Span A crosses above/below Senkou Span B
    # Twist up: Senkou A > Senkou B (bullish momentum shift)
    # Twist down: Senkou A < Senkou B (bearish momentum shift)
    twist_up = senkou_a > senkou_b
    twist_down = senkou_a < senkou_b
    
    # Align 1d EMA50 to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Discrete size to minimize fee churn
    
    # Warmup: need 52-period for Senkou B
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(twist_up[i]) or np.isnan(twist_down[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_aligned[i]
        twist_up_val = twist_up[i]
        twist_down_val = twist_down[i]
        
        if position == 0:
            # Look for entry: Ichimoku cloud twist in direction of 1d trend
            long_condition = twist_up_val and (close_val > ema_val)
            short_condition = twist_down_val and (close_val < ema_val)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: cloud twist down or price crosses below 1d EMA50
            if twist_down_val or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: cloud twist up or price crosses above 1d EMA50
            if twist_up_val or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0