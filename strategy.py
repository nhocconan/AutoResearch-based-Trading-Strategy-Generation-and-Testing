#!/usr/bin/env python3
"""
1h_4h_1d_Pivot_Reversal_v1
Hypothesis: Combines 4h trend filter (price above/below 4h EMA20) with daily pivot reversals.
Long when 4h uptrend AND price touches daily pivot support AND closes above it.
Short when 4h downtrend AND price touches daily pivot resistance AND closes below it.
Designed for low trade frequency by requiring 4h trend alignment and daily pivot rejection.
Works in bull via buying dips in uptrend, in bear via selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Pivot_Reversal_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H DATA ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    close_4hs = pd.Series(close_4h)
    ema20_4h = close_4hs.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Pivot Points (Standard)
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align daily pivots
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # 4H trend filter
        trend_up = close[i] > ema20_4h_aligned[i]
        trend_down = close[i] < ema20_4h_aligned[i]
        
        # Daily pivot rejection with confirmation
        near_support = low[i] <= s1_aligned[i] * 1.001  # Allow 0.1% slippage
        near_resistance = high[i] >= r1_aligned[i] * 0.999
        close_above_support = close[i] > s1_aligned[i]
        close_below_resistance = close[i] < r1_aligned[i]
        
        # Entry conditions
        long_setup = trend_up and near_support and close_above_support
        short_setup = trend_down and near_resistance and close_below_resistance
        
        # Exit when trend changes
        exit_long = not trend_up
        exit_short = not trend_down
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals