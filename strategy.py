#!/usr/bin/env python3
"""
12h_Pivot_Point_Breakout_With_Weekly_Trend_Filter
Hypothesis: On 12h timeframe, use daily Camarilla pivot levels (R1, S1) for breakout entries, filtered by weekly trend (price above/below weekly EMA200). Long when price breaks above R1 with weekly uptrend; short when price breaks below S1 with weekly downtrend. This captures institutional intraday levels with higher timeframe trend alignment, reducing false breakouts. Weekly EMA200 provides robust trend filter that works in both bull (avoids shorts in uptrend) and bear (avoids longs in downtrend). Targets 15-25 trades/year by requiring pivot break + trend filter, with position size 0.25.
"""

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    hl_range = high_1d - low_1d
    r1_1d = close_1d + 1.1 * hl_range / 12
    s1_1d = close_1d - 1.1 * hl_range / 12
    
    # Align pivot levels to 12h timeframe (wait for daily bar close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA to 12h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # need at least one bar of aligned data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 and weekly uptrend (price > EMA200)
            if (close[i] > r1_1d_aligned[i] and close[i] > ema200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 and weekly downtrend (price < EMA200)
            elif (close[i] < s1_1d_aligned[i] and close[i] < ema200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below S1 (reversal signal)
            if close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (reversal signal)
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_Point_Breakout_With_Weekly_Trend_Filter"
timeframe = "12h"
leverage = 1.0