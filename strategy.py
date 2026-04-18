#!/usr/bin/env python3
"""
1d_WeeklyPivot_Direction_1WeekTrendFilter_v1
Hypothesis: Use weekly pivot levels (R1/S1) for entry/exit on daily timeframe, with weekly trend filter to avoid counter-trend trades.
Long when price crosses above weekly R1 and weekly trend is up (price > weekly SMA50).
Short when price crosses below weekly S1 and weekly trend is down (price < weekly SMA50).
Exit when price returns to weekly pivot (PP) or opposite condition occurs.
Weekly trend filter reduces whipsaws in ranging markets. Pivot levels provide institutional reference points.
Target: 10-25 trades/year by requiring both pivot break and trend alignment.
Works in bull via long signals, bear via short signals, and ranges via reduced frequency.
"""

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
    
    # Get weekly data for pivot levels and trend filter
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R1 = 2*PP - Low
    # S1 = 2*PP - High
    pp_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    r1_weekly = 2.0 * pp_weekly - low_weekly
    s1_weekly = 2.0 * pp_weekly - high_weekly
    
    # Weekly trend filter: price relative to 50-period SMA
    sma_period = 50
    sma_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= sma_period:
        for i in range(sma_period, len(close_weekly)):
            sma_weekly[i] = np.mean(close_weekly[i-sma_period:i])
    
    # Align weekly data to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp_weekly)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    sma_aligned = align_htf_to_ltf(prices, df_weekly, sma_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = sma_period + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(sma_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above weekly R1 AND weekly trend is up (price > weekly SMA50)
            if close[i] > r1_aligned[i] and close[i] > sma_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below weekly S1 AND weekly trend is down (price < weekly SMA50)
            elif close[i] < s1_aligned[i] and close[i] < sma_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly pivot OR trend turns down
            if close[i] <= pp_aligned[i] or close[i] < sma_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly pivot OR trend turns up
            if close[i] >= pp_aligned[i] or close[i] > sma_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_Direction_1WeekTrendFilter_v1"
timeframe = "1d"
leverage = 1.0