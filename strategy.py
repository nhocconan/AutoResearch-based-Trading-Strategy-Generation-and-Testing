#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_WeeklyTrend_Filter
Hypothesis: Weekly Donchian(20) breakout with weekly EMA50 trend filter on daily chart.
Trades breakouts in the direction of weekly trend to capture momentum in both bull and bear markets.
Targets 10-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Donchian and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate 20-period high and low for Donchian channels
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(ema_50_weekly_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend direction from weekly EMA50
        trend_up = close[i] > ema_50_weekly_aligned[i]
        trend_down = close[i] < ema_50_weekly_aligned[i]
        
        # Breakout conditions
        long_breakout = high[i] > donchian_high_aligned[i]
        short_breakout = low[i] < donchian_low_aligned[i]
        
        # Entry logic
        long_entry = trend_up and long_breakout
        short_entry = trend_down and short_breakout
        
        # Exit logic: trend reversal
        long_exit = not trend_up
        short_exit = not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0