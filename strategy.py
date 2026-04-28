#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_Filter
Hypothesis: Daily Donchian(20) breakouts with weekly trend filter capture major trend moves.
Works in bull markets (breakouts to new highs) and bear markets (breakdowns to new lows).
Weekly trend filter avoids counter-trend trades. Target: 15-25 trades/year to minimize fee drag.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Donchian channels (20-period)
    # Use pandas rolling for efficiency
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 days for Donchian
    
    for i in range(start_idx, n):
        # Skip if weekly trend data not available
        if np.isnan(ema_20_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_uptrend = close > ema_20_1w_aligned[i]
        weekly_downtrend = close < ema_20_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_high = high[i] > donchian_high[i-1]  # Break above previous period's high
        breakdown_low = low[i] < donchian_low[i-1]    # Break below previous period's low
        
        # Entry logic: only trade in direction of weekly trend
        if breakout_high and weekly_uptrend and position <= 0:
            signals[i] = 0.30
            position = 1
        elif breakdown_low and weekly_downtrend and position >= 0:
            signals[i] = -0.30
            position = -1
        # Exit when price crosses back through the opposite Donchian level
        elif position == 1 and low[i] < donchian_low[i-1]:
            signals[i] = -0.30  # Reverse to short
            position = -1
        elif position == -1 and high[i] > donchian_high[i-1]:
            signals[i] = 0.30   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0