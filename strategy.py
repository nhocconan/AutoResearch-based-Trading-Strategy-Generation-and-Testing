#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Positioning
Hypothesis: Weekly pivot levels act as strong support/resistance zones. Price approaching S1/R1 with rejection (close away from level) and aligned daily trend (price > daily EMA50 for longs, < for shorts) offers high-probability mean-reversion entries. Works in both bull/bear markets by fading extremes within the weekly range. Uses 30% position size, targeting ~20-30 trades/year to minimize fee drag.
"""

name = "6h_Weekly_Pivot_Positioning"
timeframe = "6h"
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
    
    # Get weekly data for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2P - H, R1 = 2P - L
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    s1 = 2 * pivot - weekly_high  # Weekly support 1
    r1 = 2 * pivot - weekly_low   # Weekly resistance 1
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    
    # Daily trend filter: EMA(50) on daily close
    df_daily = get_htf_data(prices, '1d')
    ema50_daily = pd.Series(df_daily['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        if position == 0:
            # LONG setup: price near S1, rejecting level (close > S1), daily uptrend
            near_s1 = low[i] <= s1_aligned[i] * 1.005  # Within 0.5% of S1
            rejecting_s1 = close[i] > s1_aligned[i]    # Closed above S1
            daily_uptrend = close[i] > ema50_daily_aligned[i]
            
            if near_s1 and rejecting_s1 and daily_uptrend:
                signals[i] = 0.30
                position = 1
            # SHORT setup: price near R1, rejecting level (close < R1), daily downtrend
            elif (high[i] >= r1_aligned[i] * 0.995 and  # Within 0.5% of R1
                  close[i] < r1_aligned[i] and          # Closed below R1
                  close[i] < ema50_daily_aligned[i]): # Daily downtrend
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches pivot or daily trend breaks
            if (close[i] >= pivot_aligned[i] or 
                close[i] < ema50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price reaches pivot or daily trend breaks
            if (close[i] <= pivot_aligned[i] or 
                close[i] > ema50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals