#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_v1
Hypothesis: 6-hour Donchian(20) breakout filtered by weekly pivot direction.
Long when price breaks above 20-period high AND weekly pivot trend is up.
Short when price breaks below 20-period low AND weekly pivot trend is down.
Weekly pivot trend defined as: price above/below weekly VWAP (approximated via weekly close).
Uses 6h primary timeframe with 1w HTF for trend filter. Targets 50-150 total trades over 4 years.
Designed to work in bull via breakouts with trend, in bear via breakdowns with trend.
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
    
    # Get weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly trend: 1 if close > prev close (up), -1 if close < prev close (down)
    weekly_close = df_1w['close'].values
    weekly_prev_close = np.roll(weekly_close, 1)
    weekly_prev_close[0] = np.nan  # first value has no previous
    weekly_trend_raw = np.where(weekly_close > weekly_prev_close, 1,
                                np.where(weekly_close < weekly_prev_close, -1, 0))
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_raw)
    
    # Calculate Donchian channels (20-period) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of weekly trend (needs 2), Donchian (20)
    start_idx = max(2, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_trend_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        weekly_trend = int(weekly_trend_aligned[i])  # -1, 0, or 1
        
        if position == 0:
            # Long: price breaks above Donchian high AND weekly trend up
            if (close_val > highest_high[i]) and (weekly_trend == 1):
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price breaks below Donchian low AND weekly trend down
            elif (close_val < lowest_low[i]) and (weekly_trend == -1):
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: weekly trend flips down OR price retrace to midpoint
            if (weekly_trend == -1) or (close_val < (highest_high[i] + lowest_low[i]) / 2):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: weekly trend flips up OR price retrace to midpoint
            if (weekly_trend == 1) or (close_val > (highest_high[i] + lowest_low[i]) / 2):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_v1"
timeframe = "6h"
leverage = 1.0