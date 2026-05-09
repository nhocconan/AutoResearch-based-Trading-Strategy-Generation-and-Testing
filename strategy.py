#!/usr/bin/env python3
name = "12H_3wk_High_Low_Breakout_1wTrend"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and breakout levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 3-week high/low from weekly close (breakout levels)
    close_1w = df_1w['close'].values
    high_3w = pd.Series(close_1w).rolling(window=3, min_periods=3).max().values
    low_3w = pd.Series(close_1w).rolling(window=3, min_periods=3).min().values
    
    # Align to 12h
    high_3w_aligned = align_htf_to_ltf(prices, df_1w, high_3w)
    low_3w_aligned = align_htf_to_ltf(prices, df_1w, low_3w)
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    volume_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(high_3w_aligned[i]) or np.isnan(low_3w_aligned[i]) or np.isnan(ema20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 3-week high + above weekly EMA20 + volume confirmation
            if close[i] > high_3w_aligned[i] and close[i] > ema20_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 3-week low + below weekly EMA20 + volume confirmation
            elif close[i] < low_3w_aligned[i] and close[i] < ema20_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below weekly EMA20 (trend change)
            if close[i] < ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly EMA20 (trend change)
            if close[i] > ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals