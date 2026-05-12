#!/usr/bin/env python3
"""
1D_WEEKLY_HIGH_LOW_BREAKOUT_VOLUME
Hypothesis: Breakout of previous week's high/low on daily timeframe with volume confirmation.
Weekly levels provide strong support/resistance; breakouts indicate institutional interest.
Volume spike confirms validity. Works in both bull (breakout continuation) and bear (breakdown continuation).
Target: 15-25 trades/year to stay under 100 total over 4 years.
"""
name = "1D_WEEKLY_HIGH_LOW_BREAKOUT_VOLUME"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for high/low levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's high and low
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_high[0] = prev_weekly_low[0] = np.nan
    
    # Align weekly levels to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, prev_weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, prev_weekly_low)
    
    # Volume spike: current daily volume > 2x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above previous week's high with volume spike
            if high[i] > weekly_high_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below previous week's low with volume spike
            elif low[i] < weekly_low_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below previous week's low
            if close[i] < weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above previous week's high
            if close[i] > weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals