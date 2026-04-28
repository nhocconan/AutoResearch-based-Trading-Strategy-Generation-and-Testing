#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_TrendFilter
Hypothesis: Uses weekly Donchian channel (20-week period) breakouts with weekly EMA trend filter on the daily timeframe.
Trades only in the direction of the weekly trend with no volume requirement to avoid overtrading.
Designed for low trade frequency (target: 10-30 trades per year) to minimize fee drag in bear markets.
Works in both bull and bear markets by following the weekly trend direction.
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
    
    # Get weekly data for Donchian channels and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Donchian upper (20-week high)
    donch_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    # Donchian lower (20-week low)
    donch_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_weekly, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_weekly, donch_low)
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend direction
        trend_up = close[i] > ema_50_weekly_aligned[i]
        trend_down = close[i] < ema_50_weekly_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > donch_high_aligned[i]
        breakout_down = close[i] < donch_low_aligned[i]
        
        # Entry logic: Breakout in direction of weekly trend
        long_entry = breakout_up and trend_up
        short_entry = breakout_down and trend_down
        
        # Exit logic: Opposite breakout or trend reversal
        long_exit = breakout_down or not trend_up
        short_exit = breakout_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
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
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian_Breakout_TrendFilter"
timeframe = "1d"
leverage = 1.0