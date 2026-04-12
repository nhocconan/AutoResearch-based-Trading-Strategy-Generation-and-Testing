#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_volume_trend
Hypothesis: 4-hour strategy using daily Donchian breakouts with volume confirmation and 1-week trend filter.
Trades only in direction of weekly trend to avoid counter-trend whipsaws in bear markets.
Uses volume spike (1.5x average) to confirm breakout strength.
Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag.
Works in bull markets via breakouts and in bear markets via trend filter preventing false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Donchian channels (20-period)
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    for i in range(20, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-20:i])
        lowest_low[i] = np.min(low_1d[i-20:i])
    
    # Calculate 20-period average volume for spike detection
    avg_volume = np.full_like(volume_1d, np.nan)
    for i in range(20, len(volume_1d)):
        avg_volume[i] = np.mean(volume_1d[i-20:i])
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend direction
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(avg_volume_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: above EMA50 = uptrend, below = downtrend
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume spike: current volume > 1.5x average volume
        volume_spike = volume[i] > 1.5 * avg_volume_aligned[i]
        
        # Breakout conditions
        long_breakout = high[i] > highest_high_aligned[i] and volume_spike
        short_breakout = low[i] < lowest_low_aligned[i] and volume_spike
        
        # Entry logic: only trade in direction of weekly trend
        if weekly_uptrend and long_breakout and position != 1:
            position = 1
            signals[i] = 0.30  # Long 30%
        elif weekly_downtrend and short_breakout and position != -1:
            position = -1
            signals[i] = -0.30  # Short 30%
        # Exit: opposite breakout or trend change
        elif position == 1 and (short_breakout or not weekly_uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (long_breakout or not weekly_downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals

name = "4h_1d_donchian_breakout_volume_trend"
timeframe = "4h"
leverage = 1.0