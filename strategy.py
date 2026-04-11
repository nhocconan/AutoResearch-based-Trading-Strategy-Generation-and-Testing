#!/usr/bin/env python3
# 6h_1d_weekly_pivot_breakout_v1
# Strategy: 6h price breakout above/below weekly pivot levels (R1/S1) with 1d volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Weekly pivot points act as key support/resistance levels. Breakouts above R1 or below S1
# indicate institutional interest and trend continuation. Volume confirmation filters false breakouts.
# Works in bull markets via upside breakouts and bear markets via downside breakdowns.
# Target: 15-35 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "6h_1d_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week (using 1d data)
    # Need prior week's high, low, close
    # We'll compute pivot for each day based on previous 7 days
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling window of 7 days for weekly high/low/close
    weekly_high = pd.Series(high_1d).rolling(window=7, min_periods=7).max().shift(1).values
    weekly_low = pd.Series(low_1d).rolling(window=7, min_periods=7).min().shift(1).values
    weekly_close = pd.Series(close_1d).rolling(window=7, min_periods=7).last().shift(1).values
    
    # Calculate pivot and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: 1d volume > 1.5x 20-day average
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    vol_confirm = vol_1d_aligned > 1.5 * vol_avg_20_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(close[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(vol_confirm[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_long = close[i] > r1_aligned[i] and vol_confirm[i]
        breakout_short = close[i] < s1_aligned[i] and vol_confirm[i]
        
        # Entry conditions
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to pivot level (mean reversion)
        elif position == 1 and close[i] < pivot_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > pivot_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals