#!/usr/bin/env python3
# 6h_Turtle_Soup_Reversal_1dTrend
# Hypothesis: Turtle Soup reversal pattern on 6h with 1-day trend filter.
# Turtle Soup: false breakout of 20-bar high/low followed by reversal.
# Long: price makes new 20-bar low then closes above that low (stop run reversal).
# Short: price makes new 20-bar high then closes below that high.
# Uses 1-day EMA50 as trend filter to align with higher timeframe bias.
# Targets 15-25 trades/year to minimize fee drag while capturing reversal edges.

name = "6h_Turtle_Soup_Reversal_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 for trend filter (daily)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 20-bar highest high and lowest low for Turtle Soup
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_20 = high_series.rolling(window=20, min_periods=20).max().values
    lowest_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 bars for highest/lowest calculation
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Turtle Soup Long: false breakdown below 20-bar low, then reversal
            # Condition: current low <= 20-bar low (breakdown) AND close > 20-bar low (reversal)
            if (low[i] <= lowest_20[i] and close[i] > lowest_20[i] and 
                close[i] > ema50_1d_aligned[i]):  # Uptrend filter
                signals[i] = 0.25
                position = 1
            # Turtle Soup Short: false breakout above 20-bar high, then reversal
            # Condition: current high >= 20-bar high (breakout) AND close < 20-bar high (reversal)
            elif (high[i] >= highest_20[i] and close[i] < highest_20[i] and 
                  close[i] < ema50_1d_aligned[i]):  # Downtrend filter
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below the 20-bar low (failure of reversal)
            if close[i] <= lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above the 20-bar high (failure of reversal)
            if close[i] >= highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals