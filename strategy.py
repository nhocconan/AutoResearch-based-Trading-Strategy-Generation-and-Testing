#!/usr/bin/env python3
"""
1d_1w_Turtle_Reversal_v1
Hypothesis: Use weekly Donchian channels (20) for trend direction and daily Donchian breakouts (20) for entries.
Long when price breaks above daily DONCHIAN(20) high and weekly trend is up.
Short when price breaks below daily DONCHIAN(20) low and weekly trend is down.
Exit on opposite daily Donchian breakout or weekly trend reversal.
Designed for low trade frequency (10-25/year) with trend following in both bull and bear markets.
Uses volume confirmation to avoid false breakouts and ATR-based position sizing for risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Turtle_Reversal_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR TREND DIRECTION ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: price above/below midpoint of weekly channel
    mid_20 = (high_20 + low_20) / 2
    weekly_uptrend = close_1w > mid_20
    weekly_downtrend = close_1w < mid_20
    
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # === DAILY DATA FOR ENTRY SIGNALS ===
    # Daily Donchian channels (20)
    high_20d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(high_20d[i]) or np.isnan(low_20d[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_uptrend_aligned[i]) or
            np.isnan(weekly_downtrend_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Entry conditions
        long_breakout = close[i] > high_20d[i] and weekly_uptrend_aligned[i] and strong_volume
        short_breakout = close[i] < low_20d[i] and weekly_downtrend_aligned[i] and strong_volume
        
        # Exit conditions: opposite breakout or weekly trend reversal
        exit_long = position == 1 and (close[i] < low_20d[i] or not weekly_uptrend_aligned[i])
        exit_short = position == -1 and (close[i] > high_20d[i] or not weekly_downtrend_aligned[i])
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals