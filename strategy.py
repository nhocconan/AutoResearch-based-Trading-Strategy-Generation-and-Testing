#!/usr/bin/env python3
"""
6h_1w_21_EMA_Supertrend_Swing_V1
Hypothesis: Uses weekly 21 EMA trend filter and 6h Supertrend for entries. Only takes trades aligned with weekly trend.
Supertrend provides dynamic stop and entry signals. Weekly EMA filter avoids counter-trend trades in chop.
Designed for low trade frequency (target 15-30/year) with high win rate by requiring trend alignment.
Works in bull (follows uptrend) and bear (follows downtrend) by only trading in direction of weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_21_EMA_Supertrend_Swing_V1"
timeframe = "6h"
leverage = 1.0

def supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = np.zeros_like(close)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(close)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close)
    final_lb = np.zeros_like(close)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(close)):
        if close[i-1] <= final_ub[i-1]:
            final_ub[i] = min(basic_ub[i], final_ub[i-1])
        else:
            final_ub[i] = basic_ub[i]
            
        if close[i-1] >= final_lb[i-1]:
            final_lb[i] = max(basic_lb[i], final_lb[i-1])
        else:
            final_lb[i] = basic_lb[i]
    
    # Supertrend
    supertrend_val = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close)):
        if close[i] > final_ub[i-1]:
            direction[i] = 1
        elif close[i] < final_lb[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend_val[i] = final_lb[i]
        else:
            supertrend_val[i] = final_ub[i]
    
    return supertrend_val, direction, atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data ONCE before loop for EMA filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly 21 EMA for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate Supertrend on 6h data
    st, st_dir, atr = supertrend(high, low, close, period=10, multiplier=3.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Warmup for EMA and Supertrend
        # Skip if any required data is invalid
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(st[i]) or 
            np.isnan(st_dir[i]) or np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price relative to weekly 21 EMA
        above_weekly_ema = close[i] > ema_21_1w_aligned[i]
        below_weekly_ema = close[i] < ema_21_1w_aligned[i]
        
        # Supertrend direction
        uptrend = st_dir[i] == 1
        downtrend = st_dir[i] == -1
        
        # Entry conditions: Supertrend signal aligned with weekly trend
        long_entry = uptrend and above_weekly_ema
        short_entry = downtrend and below_weekly_ema
        
        # Exit conditions: Supertrend reversal or weekly trend violation
        long_exit = not uptrend or not above_weekly_ema
        short_exit = not downtrend or not below_weekly_ema
        
        # Priority: entry > exit > hold
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
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals