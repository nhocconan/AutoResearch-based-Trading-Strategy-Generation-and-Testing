#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_HTFTrend_Volume
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and 1d EMA34 trend filter + volume confirmation.
Enters long when price breaks above 20-period high with bullish weekly pivot bias (price > weekly pivot) and bullish 1d trend (price > EMA34) and volume spike (>1.5x 20 EMA volume).
Enters short when price breaks below 20-period low with bearish weekly pivot bias (price < weekly pivot) and bearish 1d trend (price < EMA34) and volume spike.
Exits on opposite Donchian breakout or when weekly pivot bias flips.
Designed for 50-150 total trades over 4 years (12-37/year) to avoid fee drag on 6h timeframe.
Uses discrete position sizing (0.25) to minimize churn. Works in both bull and bear markets by aligning with weekly pivot and 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot point (prior completed week)
    df_1w = get_htf_data(prices, '1w')
    prior_weekly_high = np.roll(df_1w['high'].values, 1)
    prior_weekly_low = np.roll(df_1w['low'].values, 1)
    prior_weekly_close = np.roll(df_1w['close'].values, 1)
    prior_weekly_high[0] = np.nan
    prior_weekly_low[0] = np.nan
    prior_weekly_close[0] = np.nan
    weekly_pivot = (prior_weekly_high + prior_weekly_low + prior_weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period)
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need weekly shift + 34-period EMA + 20-period Donchian)
    start_idx = max(1 + 34, lookback)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(highest[i]) or np.isnan(lowest[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: Donchian breakout above + bullish weekly pivot + bullish 1d trend + volume spike
        if (close[i] > highest[i] and 
            close[i] > weekly_pivot_aligned[i] and 
            close[i] > ema_34_1d_aligned[i] and 
            volume_spike[i]):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Donchian breakdown below + bearish weekly pivot + bearish 1d trend + volume spike
        elif (close[i] < lowest[i] and 
              close[i] < weekly_pivot_aligned[i] and 
              close[i] < ema_34_1d_aligned[i] and 
              volume_spike[i]):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite Donchian breakout or weekly pivot bias flip
        elif position == 1 and (close[i] < lowest[i] or close[i] < weekly_pivot_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > highest[i] or close[i] > weekly_pivot_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_HTFTrend_Volume"
timeframe = "6h"
leverage = 1.0