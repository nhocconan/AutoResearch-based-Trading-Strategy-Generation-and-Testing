#!/usr/bin/env python3
"""
1D_Equilibrium_Pivot_Bounce_1WTrend
Hypothesis: Price tends to revert to the weekly equilibrium (pivot) within strong weekly trends.
In uptrends: buy near weekly S1; in downtrends: sell near weekly R1. Uses 1D timeframe for precision.
Weekly trend filter (EMA34) ensures alignment with primary direction. Avoids counter-trend entries.
Targets 7-25 trades/year on 1D to minimize fee drag.
"""
name = "1D_Equilibrium_Pivot_Bounce_1WTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1W data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly pivot and S1/R1 levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    s1 = pivot - (range_1w * 1.1 / 6)  # S1 level
    r1 = pivot + (range_1w * 1.1 / 6)  # R1 level
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # Weekly EMA34 for trend direction
    close_1w_series = pd.Series(df_1w['close'])
    ema_34 = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 2)  # EMA34 warmup + 1 bar for crossover
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 14 days between trades (2 weeks) to reduce frequency
            if bars_since_exit < 14:
                continue
                
            # Long: price near weekly S1 in weekly uptrend
            if (low[i] <= s1_aligned[i] * 1.005 and  # Allow 0.5% tolerance
                close[i] > ema_34_aligned[i] and      # Weekly uptrend
                close[i] > close[i-1]):               # Confirmation: close above prior close
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price near weekly R1 in weekly downtrend
            elif (high[i] >= r1_aligned[i] * 0.995 and  # Allow 0.5% tolerance
                  close[i] < ema_34_aligned[i] and      # Weekly downtrend
                  close[i] < close[i-1]):               # Confirmation: close below prior close
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price crosses weekly pivot (mean reversion complete)
            if position == 1 and close[i] >= pivot[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] <= pivot[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals