#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot and Volume Confirmation
Hypothesis: Donchian breakouts on 6h timeframe capture momentum in trending markets.
Weekly pivot (from 1d data) provides directional bias: trade only breakouts in direction of weekly trend.
Volume confirms institutional participation. Works in bull (breakouts above weekly pivot) and bear (breakdowns below weekly pivot).
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for weekly pivot calculation (once before loop)
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Weekly pivot calculation (using prior week's daily data)
    # Calculate weekly high/low/close from daily data (simplified: use last 5 days)
    def calculate_weekly_pivot(high, low, close):
        # For each point, use prior 5-day weekly aggregate
        # Weekly high = max of last 5 daily highs
        weekly_high = pd.Series(high).rolling(window=5, min_periods=5).max().values
        # Weekly low = min of last 5 daily lows
        weekly_low = pd.Series(low).rolling(window=5, min_periods=5).min().values
        # Weekly close = last daily close
        weekly_close = close
        # Pivot point = (weekly_high + weekly_low + weekly_close) / 3
        pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        return weekly_high, weekly_low, pivot
    
    weekly_high, weekly_low, weekly_pivot = calculate_weekly_pivot(high_daily, low_daily, close_daily)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_daily, weekly_pivot)
    weekly_high_aligned = align_htf_to_ltf(prices, df_daily, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_daily, weekly_low)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20 period)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.2 * vol_ma)  # Require above average volume
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(donchian_period, 5) + 14
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below weekly pivot OR stoploss
            if (close[i] <= weekly_pivot_aligned[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above weekly pivot OR stoploss
            if (close[i] >= weekly_pivot_aligned[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout in direction of weekly trend + volume
            # Weekly trend: price above/below weekly pivot
            weekly_uptrend = close[i] > weekly_pivot_aligned[i]
            weekly_downtrend = close[i] < weekly_pivot_aligned[i]
            
            long_setup = (close[i] > donchian_high[i] and weekly_uptrend and vol_filter[i])
            short_setup = (close[i] < donchian_low[i] and weekly_downtrend and vol_filter[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals