#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Breakout_With_Trend_Filter
Hypothesis: Weekly pivot points (R1/S1) derived from weekly high/low/close act as strong support/resistance.
Breakouts above weekly R1 or below S1 with daily trend alignment (EMA20) capture momentum moves.
Exit on reversion to weekly pivot point (PP). Position size 0.25 targets ~15-25 trades/year.
Works in both bull (breakouts with trend) and bear (mean reversion at extremes) via trend filter.
"""

name = "1d_Weekly_Pivot_Breakout_With_Trend_Filter"
timeframe = "1d"
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    weekly_pp = (h_1w + l_1w + c_1w) / 3.0
    weekly_r1 = c_1w + (h_1w - l_1w) * 1.1 / 12.0
    weekly_s1 = c_1w - (h_1w - l_1w) * 1.1 / 12.0
    
    # Align weekly pivots to daily chart (wait for weekly close)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Daily trend filter: EMA20
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above weekly R1 with uptrend
            if close[i] > weekly_r1_aligned[i] and close[i] > ema20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below weekly S1 with downtrend
            elif close[i] < weekly_s1_aligned[i] and close[i] < ema20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly pivot or trend reverses
            if close[i] < weekly_pp_aligned[i] or close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly pivot or trend reverses
            if close[i] > weekly_pp_aligned[i] or close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals