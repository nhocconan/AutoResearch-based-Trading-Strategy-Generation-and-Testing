#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_TrendFilter_v1
Hypothesis: Weekly pivot levels provide strong institutional support/resistance. Donchian(20) breakouts on 6f with volume confirmation and weekly pivot alignment capture swing continuations. Weekly trend filter (price vs weekly EMA20) ensures alignment with higher timeframe momentum. Works in both bull/bear markets by trading breakouts in direction of weekly trend. Target: 60-120 total trades over 4 years (15-30/year).
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
    
    # Load weekly data ONCE before loop for HTF trend filter and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Calculate weekly pivot levels (using previous week's OHLC)
    # Standard pivot: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    if len(df_1w) < 2:
        return np.zeros(n)
    
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Weekly pivot calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Donchian(20) channels on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_20_aligned[i]
        downtrend = close[i] < ema_20_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = high[i] > highest_high[i-1]  # Current high > previous period's highest high
        breakout_down = low[i] < lowest_low[i-1]   # Current low < previous period's lowest low
        
        # Long logic: Donchian breakout up + volume spike + price above weekly pivot + weekly uptrend
        if breakout_up and volume_spike[i] and close[i] > r1_aligned[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: Donchian breakout down + volume spike + price below weekly support + weekly downtrend
        elif breakout_down and volume_spike[i] and close[i] < s1_aligned[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: reversal breakout or trend change
        elif position == 1 and (breakout_down or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (breakout_up or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0