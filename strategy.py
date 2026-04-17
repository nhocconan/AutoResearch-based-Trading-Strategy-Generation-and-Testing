#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + Weekly Trend Filter.
Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with EMA13 on 6h.
Weekly trend from 1w EMA34: bullish when close > EMA34, bearish when close < EMA34.
Long when Bull Power > 0 and weekly trend bullish; Short when Bear Power > 0 and weekly trend bearish.
Exit when power reverses or weekly trend changes.
Uses discrete position sizing (0.25) to minimize fee churn.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Get 1w data for weekly trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    close_1w_s = pd.Series(close_1w)
    ema34_1w = close_1w_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Weekly trend: bullish when close > EMA34, bearish when close < EMA34
    weekly_bullish = close_1w > ema34_1w
    weekly_bearish = close_1w < ema34_1w
    
    # Align weekly trend to 6h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        bp = bull_power[i]
        br = bear_power[i]
        w_bull = weekly_bullish_aligned[i] > 0.5
        w_bear = weekly_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Long: Bull Power > 0 and weekly trend bullish
            if bp > 0 and w_bull:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 and weekly trend bearish
            elif br > 0 and w_bear:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 or weekly trend turns bearish
            if bp <= 0 or not w_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power <= 0 or weekly trend turns bullish
            if br <= 0 or not w_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_WeeklyTrend"
timeframe = "6h"
leverage = 1.0