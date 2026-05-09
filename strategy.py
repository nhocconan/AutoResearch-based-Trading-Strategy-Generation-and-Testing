#!/usr/bin/env python3
# 12h_Weekly_Range_Reversion
# Hypothesis: On 12h timeframe, price tends to revert to weekly range extremes (high/low) after touching them.
# Strategy: Identify weekly high and low from prior week. When price touches weekly high with bearish momentum (price < open), go short.
# When price touches weekly low with bullish momentum (price > open), go long. Exit when price returns to weekly midpoint.
# Uses weekly range as support/resistance in ranging markets, works in both bull and bear by fading extremes.
# Target: 80-120 total trades over 4 years (20-30/year) with size 0.25.

name = "12h_Weekly_Range_Reversion"
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
    open_price = prices['open'].values
    
    # Get weekly data (prior week's high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior week's high and low (shifted by 1 to avoid look-ahead)
    weekly_high = df_1w['high'].shift(1).values
    weekly_low = df_1w['low'].shift(1).values
    weekly_mid = (weekly_high + weekly_low) / 2
    
    # Align weekly levels to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_mid_aligned = align_htf_to_ltf(prices, df_1w, weekly_mid)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_mid_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price touches weekly low with bullish momentum (close > open)
            if (low[i] <= weekly_low_aligned[i] and close[i] > open_price[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches weekly high with bearish momentum (close < open)
            elif (high[i] >= weekly_high_aligned[i] and close[i] < open_price[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly midpoint or above
            if close[i] >= weekly_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly midpoint or below
            if close[i] <= weekly_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals