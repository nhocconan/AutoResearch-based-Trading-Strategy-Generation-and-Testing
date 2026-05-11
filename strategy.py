#!/usr/bin/env python3
"""
12h_1d_SuperTrend_TrendFollow
Hypothesis: Supertrend indicator on daily timeframe identifies strong trends. 
Enter long when price crosses above Supertrend on 12h chart with daily uptrend confirmation.
Enter short when price crosses below Supertrend on 12h chart with daily downtrend confirmation.
Uses ATR-based dynamic stop and re-entry mechanism to capture trends while minimizing whipsaws.
Works in both bull (riding uptrends) and bear (riding downtrends) markets by following the daily trend.
"""

name = "12h_1d_SuperTrend_TrendFollow"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Supertrend on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_multiplier = 3.0
    
    # Basic upper and lower bands
    basic_ub = (high_1d + low_1d) / 2 + atr_multiplier * atr
    basic_lb = (high_1d + low_1d) / 2 - atr_multiplier * atr
    
    # Initialize final bands
    final_ub = np.full_like(close_1d, np.nan)
    final_lb = np.full_like(close_1d, np.nan)
    
    # Calculate final bands
    for i in range(len(close_1d)):
        if np.isnan(atr[i]):
            final_ub[i] = np.nan
            final_lb[i] = np.nan
        elif i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if close_1d[i-1] <= final_ub[i-1]:
                final_ub[i] = min(basic_ub[i], final_ub[i-1])
            else:
                final_ub[i] = basic_ub[i]
                
            if close_1d[i-1] >= final_lb[i-1]:
                final_lb[i] = max(basic_lb[i], final_lb[i-1])
            else:
                final_lb[i] = basic_lb[i]
    
    # Determine Supertrend direction
    supertrend = np.full_like(close_1d, np.nan)
    trend_dir = np.full_like(close_1d, np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close_1d)):
        if np.isnan(final_ub[i]) or np.isnan(final_lb[i]):
            supertrend[i] = np.nan
            trend_dir[i] = np.nan
        elif i == 0:
            supertrend[i] = final_ub[i]
            trend_dir[i] = -1  # Start in downtrend
        else:
            if supertrend[i-1] == final_ub[i-1]:
                if close_1d[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                    trend_dir[i] = -1
                else:
                    supertrend[i] = final_lb[i]
                    trend_dir[i] = 1
            else:
                if close_1d[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                    trend_dir[i] = 1
                else:
                    supertrend[i] = final_ub[i]
                    trend_dir[i] = -1
    
    # Align Supertrend and trend direction to 12h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    trend_dir_aligned = align_htf_to_ltf(prices, df_1d, trend_dir)
    
    # Price crossover detection on 12h chart
    price_above_st = close > supertrend_aligned
    price_below_st = close < supertrend_aligned
    
    # Previous state for crossover detection
    price_above_st_prev = np.concatenate([[False], price_above_st[:-1]])
    price_below_st_prev = np.concatenate([[False], price_below_st[:-1]])
    
    # Bullish crossover: price crosses above Supertrend
    bullish_cross = price_above_st & ~price_above_st_prev
    # Bearish crossover: price crosses below Supertrend
    bearish_cross = price_below_st & ~price_below_st_prev
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if required data is NaN
        if np.isnan(supertrend_aligned[i]) or np.isnan(trend_dir_aligned[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish crossover + daily uptrend
            if bullish_cross[i] and trend_dir_aligned[i] >= 1:
                signals[i] = 0.25
                position = 1
            # Short: bearish crossover + daily downtrend
            elif bearish_cross[i] and trend_dir_aligned[i] <= -1:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses back through Supertrend OR trend reversal
            if position == 1:
                # Exit long: price crosses below Supertrend OR daily trend turns down
                if bearish_cross[i] or trend_dir_aligned[i] <= -1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above Supertrend OR daily trend turns up
                if bullish_cross[i] or trend_dir_aligned[i] >= 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals