#!/usr/bin/env python3
# 4h_52WeekLow_HighBreakout_Target_Volume_20
# Hypothesis: Price breaking above the 52-week high (or below 52-week low) with volume confirmation
# captures the start of a new major trend. The 52-week extreme acts as a strong support/resistance level.
# We use a 20-bar target to take partial profits, reducing exposure and increasing win rate.
# Works in bull (breakouts above 52w high) and bear (breakdowns below 52w low).
# Target ~20-40 trades/year to avoid fee drag.

name = "4h_52WeekLow_HighBreakout_Target_Volume_20"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 260:  # Need ~1 year of 4h data for 52-week calculation
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for 52-week high/low calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 52-week high and low (52 weeks of weekly data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 52-week high: rolling max of high over 52 weeks
    week_high_52 = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    # 52-week low: rolling min of low over 52 weeks
    week_low_52 = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    
    # Align weekly 52-week levels to 4h chart (wait for weekly close)
    week_high_52_aligned = align_htf_to_ltf(prices, df_1w, week_high_52)
    week_low_52_aligned = align_htf_to_ltf(prices, df_1w, week_low_52)
    
    # Volume confirmation: current volume > 2.0x 20-period average (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(260, n):  # Start after 52-week warmup
        if position == 0:
            # LONG: Breakout above 52-week high with volume confirmation
            if close[i] > week_high_52_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below 52-week low with volume confirmation
            elif close[i] < week_low_52_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Take partial profit at 20% gain or reverse if breaks below 52-week low
            # We don't track entry price exactly, so use close-based rules:
            # Exit if price drops back below 52-week level (invalidates breakout)
            if close[i] < week_high_52_aligned[i]:  # Broke back below breakout level
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Take partial profit at 20% gain or reverse if breaks above 52-week high
            if close[i] > week_low_52_aligned[i]:  # Broke back above breakdown level
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals