#!/usr/bin/env python3
name = "12h_1w_SuperTrend_1d_SuperTrend_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def supertrend(high, low, close, atr_period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    tr1 = pd.Series(high).rolling(2).max() - pd.Series(low).rolling(2).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean()
    
    hl2 = (pd.Series(high) + pd.Series(low)) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    
    for i in range(len(close)):
        if i == 0:
            supertrend.iloc[i] = 0
            direction.iloc[i] = 1
        else:
            if close.iloc[i] > upperband.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lowerband.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
                if direction.iloc[i] == 1 and lowerband.iloc[i] < lowerband.iloc[i-1]:
                    lowerband.iloc[i] = lowerband.iloc[i-1]
                if direction.iloc[i] == -1 and upperband.iloc[i] > upperband.iloc[i-1]:
                    upperband.iloc[i] = upperband.iloc[i-1]
            
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lowerband.iloc[i]
            else:
                supertrend.iloc[i] = upperband.iloc[i]
    
    return supertrend.values, direction.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data for primary trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Supertrend
    st_1w, dir_1w = supertrend(
        df_1w['high'].values, 
        df_1w['low'].values, 
        df_1w['close'].values,
        atr_period=10,
        multiplier=3.0
    )
    st_1w_aligned = align_htf_to_ltf(prices, df_1w, st_1w)
    dir_1w_aligned = align_htf_to_ltf(prices, df_1w, dir_1w.astype(float))
    
    # Load 1d data for entry timing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Supertrend for entry signals
    st_1d, dir_1d = supertrend(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values,
        atr_period=10,
        multiplier=3.0
    )
    st_1d_aligned = align_htf_to_ltf(prices, df_1d, st_1d)
    dir_1d_aligned = align_htf_to_ltf(prices, df_1d, dir_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(st_1w_aligned[i]) or np.isnan(dir_1w_aligned[i]) or 
            np.isnan(st_1d_aligned[i]) or np.isnan(dir_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Primary trend filter: only trade in direction of 1w Supertrend
        long_allowed = dir_1w_aligned[i] > 0
        short_allowed = dir_1w_aligned[i] < 0
        
        if position == 0:
            # Enter long: 1d Supertrend turns up + aligned with weekly trend
            if long_allowed and dir_1d_aligned[i] > 0 and dir_1d_aligned[i-1] <= 0:
                signals[i] = 0.25
                position = 1
            # Enter short: 1d Supertrend turns down + aligned with weekly trend
            elif short_allowed and dir_1d_aligned[i] < 0 and dir_1d_aligned[i-1] >= 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: 1d Supertrend turns down
            if dir_1d_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 1d Supertrend turns up
            if dir_1d_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals