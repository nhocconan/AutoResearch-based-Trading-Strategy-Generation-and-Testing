#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike
Hypothesis: Daily Donchian(20) breakouts with weekly trend filter (EMA50) and volume spike confirmation.
Only trade in direction of weekly trend to avoid counter-trend whipsaws. Volume spike confirms institutional participation.
Discrete sizing 0.25 to manage risk and minimize fee churn. Target: 15-25 trades/year.
Works in bull markets (breakouts continue) and bear markets (only trend-following breakouts reduce false signals).
"""

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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), volume MA (20), and weekly EMA50 (50 weeks -> ~350 days)
    start_idx = max(20, 20, 350)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND weekly trend bullish (close > EMA50) AND volume spike
            long_setup = (close[i] > highest_high[i]) and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below Donchian lower band AND weekly trend bearish (close < EMA50) AND volume spike
            short_setup = (close[i] < lowest_low[i]) and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel OR weekly trend turns bearish
            if (close[i] < highest_high[i] and close[i] > lowest_low[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR weekly trend turns bullish
            if (close[i] < highest_high[i] and close[i] > lowest_low[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0