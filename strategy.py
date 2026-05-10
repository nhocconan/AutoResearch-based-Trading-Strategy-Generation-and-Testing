#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_Pivot_Trend
Hypothesis: Daily price crossing above/below weekly Camarilla R1/S1 levels in direction of weekly EMA34 trend.
Uses weekly trend filter to avoid counter-trend trades, with daily execution for better timing.
Targets 10-25 trades/year on 1d to minimize fee drag while capturing institutional level breaks.
Works in bull/bear by following higher timeframe trend.
"""

name = "1d_Weekly_Camarilla_Pivot_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate weekly Camarilla levels (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_r1 = close_1w + (high_1w - low_1w) * 1.1 / 12
    camarilla_s1 = close_1w - (high_1w - low_1w) * 1.1 / 12
    
    # Align weekly Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Get daily price
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34 weeks) and enough history
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above weekly EMA34 (uptrend) AND price breaks above weekly R1
            if close[i] > ema_34_aligned[i] and high[i] > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: below weekly EMA34 (downtrend) AND price breaks below weekly S1
            elif close[i] < ema_34_aligned[i] and low[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly S1 OR trend turns bearish
            if low[i] < s1_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly R1 OR trend turns bullish
            if high[i] > r1_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals