#!/usr/bin/env python3
"""
6h_1d_Supertrend_Follow_Trend_v1
Hypothesis: Use Supertrend on daily timeframe to determine trend direction, then enter on 6h pullbacks in the direction of the daily trend.
Long when daily trend is up and 6h price pulls back to EMA20; short when daily trend is down and 6h price pulls back to EMA20.
Exit when price crosses EMA20 in the opposite direction or daily trend flips.
Designed for low trade frequency (target: 50-150 total over 4 years) by requiring trend alignment and pullback entries.
Works in bull via trend-following longs, in bear via trend-following shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Supertrend_Follow_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Supertrend on daily (ATR=10, multiplier=3)
    atr_period = 10
    multiplier = 3
    
    hl2 = (df_1d['high'] + df_1d['low']) / 2
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = abs(df_1d['low'] - df_1d['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean()
    
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = pd.Series(index=df_1d.index, dtype=float)
    direction = pd.Series(index=df_1d.index, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend.iloc[0] = upperband.iloc[0]
    direction.iloc[0] = 1
    
    for i in range(1, len(df_1d)):
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
    
    # Align daily Supertrend and direction to 6h timeframe
    supertrend_array = supertrend.values
    direction_array = direction.values
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend_array)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction_array)
    
    # EMA20 on 6h for pullback entries
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(ema20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend direction from daily Supertrend
        daily_uptrend = direction_aligned[i] == 1
        daily_downtrend = direction_aligned[i] == -1
        
        # Entry conditions: pullback to EMA20 in direction of daily trend
        long_entry = daily_uptrend and close[i] <= ema20[i] and low[i] <= ema20[i]
        short_entry = daily_downtrend and close[i] >= ema20[i] and high[i] >= ema20[i]
        
        # Exit conditions: trend flip or price crosses EMA20 in opposite direction
        long_exit = not daily_uptrend or close[i] >= ema20[i]
        short_exit = not daily_downtrend or close[i] <= ema20[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals