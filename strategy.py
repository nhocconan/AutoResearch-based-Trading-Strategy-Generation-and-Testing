#!/usr/bin/env python3
"""
6h_1w_1d_Elder_Ray_Regime_v1
Hypothesis: 6h timeframe with Elder Ray (bull/bear power) from 1d and weekly regime filter.
Only takes long when weekly trend is up (price > weekly EMA50) and daily bull power > 0.
Only takes short when weekly trend is down (price < weekly EMA50) and daily bear power < 0.
Uses 13-period EMA for power calculation. Designed to avoid whipsaws in ranging markets
by requiring alignment with weekly trend, reducing false signals during 2022-2024 chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Elder_Ray_Regime_v1"
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
    
    # Load weekly data ONCE for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Load daily data ONCE for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend regime
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily EMA13 for Elder Ray power calculation
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray components (daily)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power = high - ema_13_1d
    bear_power = low - ema_13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ema_13_1d_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend regime
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Daily Elder Ray signals
        strong_bull = bull_power_aligned[i] > 0
        strong_bear = bear_power_aligned[i] < 0
        
        # Entry conditions: aligned with weekly trend and strong daily power
        long_entry = weekly_uptrend and strong_bull
        short_entry = weekly_downtrend and strong_bear
        
        # Exit conditions: loss of weekly trend alignment or power reversal
        long_exit = (not weekly_uptrend) or (not strong_bull)
        short_exit = (not weekly_downtrend) or (not strong_bear)
        
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