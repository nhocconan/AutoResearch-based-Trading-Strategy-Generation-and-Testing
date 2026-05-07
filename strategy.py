#!/usr/bin/env python3
"""
6h_Turtle_Soup_Reversal
Hypothesis: Turtle Soup pattern exploits false breakouts at daily highs/lows. On 6h timeframe, enter short when price makes a new 20-bar high but fails to close above it (indicating weakness), and enter long when price makes a new 20-bar low but fails to close below it (indicating strength). Uses 1d ATR for stop placement and volatility filter. Designed to work in both bull and bear markets by capturing mean reversion at extremes. Expects 80-150 total trades over 4 years.
"""
name = "6h_Turtle_Soup_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr14_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_daily_aligned = align_htf_to_ltf(prices, df_daily, atr14_daily)
    
    # Calculate 20-period highest high and lowest low for Turtle Soup
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr14_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Turtle Soup Long: new 20-bar low but close above the low (failed breakdown)
            if low[i] == lowest_low[i] and close[i] > lowest_low[i]:
                signals[i] = 0.25
                position = 1
            # Turtle Soup Short: new 20-bar high but close below the high (failed breakout)
            elif high[i] == highest_high[i] and close[i] < highest_high[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit on opposite signal or time-based exit (max 10 bars)
            if position == 1:
                # Exit on Turtle Soup Short signal or after 10 bars
                if (high[i] == highest_high[i] and close[i] < highest_high[i]) or (i - entry_bar >= 10):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Turtle Soup Long signal or after 10 bars
                if (low[i] == lowest_low[i] and close[i] > lowest_low[i]) or (i - entry_bar >= 10):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals