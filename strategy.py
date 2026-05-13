#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Donchian_Breakout_Trend_Volume
Hypothesis: Weekly pivot points define key support/resistance zones. Donchian(20) breakouts
in the direction of weekly trend (price vs weekly pivot) with volume confirmation capture
institutional flow. Works in bull/bear by using weekly pivot as dynamic bias filter.
Target: 15-30 trades/year per symbol to avoid fee drag.
"""

name = "6h_Weekly_Pivot_Donchian_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly pivot points (standard calculation)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    
    # Weekly trend: price above/below weekly pivot
    weekly_trend_up = weekly_close > pp
    weekly_trend_down = weekly_close < pp
    
    # Align weekly data to 6h timeframe (wait for weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float))
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if position == 0:
            # LONG: Donchian breakout above resistance, weekly trend up, volume
            if (high[i] > highest_high[i] and 
                weekly_trend_up_aligned[i] > 0.5 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Donchian breakdown below support, weekly trend down, volume
            elif (low[i] < lowest_low[i] and 
                  weekly_trend_down_aligned[i] > 0.5 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price falls back below Donchian midpoint or weekly trend turns
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (close[i] < midpoint or weekly_trend_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above Donchian midpoint or weekly trend turns
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (close[i] > midpoint or weekly_trend_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals