#!/usr/bin/env python3
"""
4h_12h_hlc_hybrid
Hybrid strategy using 12h price action (higher highs/lows) for trend direction,
4h Donchian breakout for entry timing, and volume confirmation.
Designed for low trade frequency (<40/year) to minimize fee drag.
Works in both bull and bear markets by following higher timeframe structure.
"""

name = "4h_12h_hlc_hybrid"
timeframe = "4h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get 12h data for trend structure (higher highs/lows)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Identify swing points on 12h
    # Higher High: current high > previous high AND previous high > high before that
    hh_12h = np.zeros(len(df_12h), dtype=bool)
    ll_12h = np.zeros(len(df_12h), dtype=bool)
    for i in range(2, len(df_12h)):
        hh_12h[i] = (high_12h[i] > high_12h[i-1]) and (high_12h[i-1] > high_12h[i-2])
        ll_12h[i] = (low_12h[i] < low_12h[i-1]) and (low_12h[i-1] < low_12h[i-2])
    
    # Trend state: 1 = uptrend (last was HH), -1 = downtrend (last was LL), 0 = unclear
    trend_state = np.zeros(len(df_12h), dtype=int)
    current_trend = 0
    for i in range(len(df_12h)):
        if hh_12h[i]:
            current_trend = 1
        elif ll_12h[i]:
            current_trend = -1
        trend_state[i] = current_trend
    
    # Align trend state to 4h
    trend_aligned = align_htf_to_ltf(prices, df_12h, trend_state.astype(float))
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if trend data not ready
        if np.isnan(trend_aligned[i]):
            signals[i] = 0.0
            continue
        
        trend = int(trend_aligned[i])
        
        # Long entry: uptrend on 12h + price breaks above 4h Donchian high + volume
        if trend == 1 and close[i] > highest_high[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: downtrend on 12h + price breaks below 4h Donchian low + volume
        elif trend == -1 and close[i] < lowest_low[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: trend change or price retracement to midpoint
        elif position == 1 and (trend == -1 or close[i] < (highest_high[i] + lowest_low[i]) * 0.5):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (trend == 1 or close[i] > (highest_high[i] + lowest_low[i]) * 0.5):
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals