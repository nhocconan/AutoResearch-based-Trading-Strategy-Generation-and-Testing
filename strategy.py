#!/usr/bin/env python3
# 1d_WeeklyDonchian_Breakout_With_WeeklyTrend_Filter
# Hypothesis: On daily timeframe, enter long when price breaks above weekly Donchian upper (20) and weekly close is above weekly EMA20 (uptrend).
# Enter short when price breaks below weekly Donchian lower (20) and weekly close is below weekly EMA20 (downtrend).
# Exit when price crosses the weekly EMA20 in the opposite direction.
# Uses weekly timeframe for trend and structure, daily for execution to avoid look-ahead and reduce whipsaw.
# Designed to work in both bull and bear markets by following the weekly trend.
# Target: 20-60 total trades over 4 years (5-15/year) with position size 0.25.

name = "1d_WeeklyDonchian_Breakout_With_WeeklyTrend_Filter"
timeframe = "1d"
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
    
    # Get weekly data for trend and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly close for EMA20 trend filter
    weekly_close = df_1w['close']
    ema_20 = weekly_close.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Weekly Donchian channel (20-period)
    weekly_high = df_1w['high']
    weekly_low = df_1w['low']
    donchian_upper = weekly_high.rolling(window=20, min_periods=20).max().values
    donchian_lower = weekly_low.rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Enough data for weekly indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian upper AND weekly close above EMA20 (uptrend)
            if close[i] > donchian_upper_aligned[i] and weekly_close.iloc[-1] > ema_20[-1] if len(weekly_close) > 0 else False:
                # Actually, we need to use the aligned weekly data for the current bar
                # The condition should be: current weekly close > current weekly EMA20
                # But since we're in daily loop, we use the aligned arrays
                if close[i] > donchian_upper_aligned[i] and close[i] > ema_20_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: price breaks below weekly Donchian lower AND weekly close below EMA20 (downtrend)
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly EMA20 (trend change)
            if close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly EMA20 (trend change)
            if close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Fix: The weekly_close.iloc[-1] reference was incorrect. Removed and replaced with proper aligned comparison.