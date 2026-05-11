#!/usr/bin/env python3
# 6h_Supertrend_1w_Trend_Filter
# Hypothesis: Use Supertrend on 6h for entry signals, filtered by 1w Supertrend direction.
# Long when 1w Supertrend is bullish and 6h Supertrend turns bullish; short when 1w Supertrend is bearish and 6h Supertrend turns bearish.
# The 1w Supertrend acts as a regime filter to avoid counter-trend trades during strong weekly trends.
# Works in bull markets by riding uptrends and in bear markets by riding downtrends, avoiding whipsaws via higher timeframe filter.
# Uses ATR-based Supertrend for dynamic support/resistance and trend detection.

name = "6h_Supertrend_1w_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend values, direction (1 for uptrend, -1 for downtrend)
    """
    # Calculate ATR
    tr1 = pd.Series(high).subtract(pd.Series(low)).abs()
    tr2 = pd.Series(high).subtract(pd.Series(close).shift(1)).abs()
    tr3 = pd.Series(low).subtract(pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Calculate upper and lower bands
    hl_avg = (pd.Series(high) + pd.Series(low)) / 2
    upper_band = (hl_avg + multiplier * atr).values
    lower_band = (hl_avg - multiplier * atr).values
    
    # Initialize Supertrend and direction
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # Start with uptrend assumption
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for Supertrend regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough data for Supertrend
        return np.zeros(n)
    
    # 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # --- 1w Supertrend for regime filter ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    supertrend_1w, direction_1w = calculate_supertrend(high_1w, low_1w, close_1w, period=10, multiplier=3.0)
    # Direction: 1 for uptrend, -1 for downtrend
    direction_1w_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    
    # --- 6h Supertrend for entry signals ---
    supertrend_6h, direction_6h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    direction_6h_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), direction_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Supertrend calculation
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(direction_1w_aligned[i]) or
            np.isnan(direction_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter from 1w Supertrend direction
        weekly_uptrend = direction_1w_aligned[i] == 1
        weekly_downtrend = direction_1w_aligned[i] == -1
        
        # Entry signal from 6h Supertrend direction change
        if i > 0 and not np.isnan(direction_6h_aligned[i-1]):
            # Bullish crossover: 6h Supertrend turns bullish
            bullish_crossover = (direction_6h_aligned[i-1] == -1) and (direction_6h_aligned[i] == 1)
            # Bearish crossover: 6h Supertrend turns bearish
            bearish_crossover = (direction_6h_aligned[i-1] == 1) and (direction_6h_aligned[i] == -1)
        else:
            bullish_crossover = False
            bearish_crossover = False
        
        if position == 0:
            if weekly_uptrend and bullish_crossover:
                # Long: weekly uptrend + 6h Supertrend turns bullish
                signals[i] = 0.25
                position = 1
            elif weekly_downtrend and bearish_crossover:
                # Short: weekly downtrend + 6h Supertrend turns bearish
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: 6h Supertrend turns bearish (regardless of weekly trend)
                if bearish_crossover:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 6h Supertrend turns bullish (regardless of weekly trend)
                if bullish_crossover:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals