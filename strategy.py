#!/usr/bin/env python3
# Hypothesis: 12h timeframe with daily pivot levels (S1/R1) for mean reversion entries and weekly trend filter.
# Uses daily Camarilla levels (S1/R1) for mean reversion entries and weekly EMA50 for trend filter.
# In range-bound markets, price tends to revert to mean from extreme levels (S1/R1).
# Weekly trend filter ensures we only take mean-reversion trades against the weekly trend
# (sell at R1 in weekly downtrend, buy at S1 in weekly uptrend) to avoid catching falling knives.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_Camarilla_S1_R1_1wEMA50_MeanRev"
timeframe = "12h"
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
    
    # Calculate daily Camarilla levels (S1, R1) from previous day
    prev_close = np.roll(close, 2)  # 2 bars = 1 day * 2 bars per 12h
    prev_high = np.roll(high, 2)
    prev_low = np.roll(low, 2)
    prev_close[:2] = np.nan  # First values invalid
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 4
    s1 = prev_close - 1.1 * camarilla_range / 4
    
    # Mean reversion conditions: price touches or exceeds the level
    touch_up = high >= r1
    touch_down = low <= s1
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly trend: price above/below EMA50
    weekly_uptrend = close > ema_50_1w_aligned
    weekly_downtrend = close < ema_50_1w_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(touch_up[i]) or np.isnan(touch_down[i]) or
            np.isnan(weekly_uptrend[i]) or np.isnan(weekly_downtrend[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion long: price touches S1 + weekly uptrend (buy weakness in uptrend) + volume
            if touch_down[i] and weekly_uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Mean reversion short: price touches R1 + weekly downtrend (sell strength in downtrend) + volume
            elif touch_up[i] and weekly_downtrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to previous day's close or weekly trend changes
            if close[i] >= prev_close[i] or not weekly_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to previous day's close or weekly trend changes
            if close[i] <= prev_close[i] or not weekly_downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals