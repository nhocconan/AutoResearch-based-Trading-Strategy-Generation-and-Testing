#1d_HighLowBreakout_with_1wTrend

#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using weekly EMA trend filter and daily high/low breakout.
In weekly uptrend (price > weekly EMA50), buy when price breaks above previous day's high.
In weekly downtrend (price < weekly EMA50), sell when price breaks below previous day's low.
Uses previous day's high/low for structure (avoids lookahead) and weekly EMA for trend.
Targets 10-20 trades/year (40-80 total over 4 years) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for previous day's high/low (structure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high and low (no lookahead)
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    
    # Align to daily timeframe (already aligned, just need to handle NaN from shift)
    prev_high_aligned = prev_high  # Already on daily index
    prev_low_aligned = prev_low    # Already on daily index
    
    # Load weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily timeframe (wait for weekly bar to close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous day data
        # Skip if indicators not ready
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        weekly_trend = ema_50_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above previous day's high + weekly uptrend
            if price_close > prev_high_aligned[i] and price_close > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below previous day's low + weekly downtrend
            elif price_close < prev_low_aligned[i] and price_close < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back below/above previous day's level in opposite direction
            if position == 1 and price_close < prev_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > prev_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_HighLowBreakout_with_1wTrend"
timeframe = "1d"
leverage = 1.0