#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyTrend_Pullback_VolumeSpike
Hypothesis: Combines 6h Donchian breakout with weekly trend filter and volume confirmation.
In trending markets (price above/below weekly 20-bar SMA), pullbacks to the 6h Donchian
channel midpoint offer high-probability entries. Volume spike filters false breakouts.
Works in both bull (buy pullbacks in uptrend) and bear (sell pullbacks in downtrend).
Target: 20-40 trades/year.
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    
    # 6h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Volume spike: >1.5x 20-period MA on 6h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma20_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly SMA20
        uptrend = close[i] > sma20_1w_aligned[i]
        downtrend = close[i] < sma20_1w_aligned[i]
        
        # Donchian breakout with pullback to midpoint
        breakout_up = (high[i] > highest_high[i-1])  # broke above prior period high
        breakout_down = (low[i] < lowest_low[i-1])   # broke below prior period low
        pullback_to_mid = abs(close[i] - donchian_mid[i]) < (highest_high[i] - lowest_low[i]) * 0.1
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Entry logic
        long_entry = uptrend and breakout_up and pullback_to_mid and vol_confirm
        short_entry = downtrend and breakout_down and pullback_to_mid and vol_confirm
        
        # Exit: opposite Donchian breakout
        long_exit = breakout_down
        short_exit = breakout_up
        
        if (long_entry or (position == 1 and not long_exit)) and position <= 0:
            signals[i] = 0.25
            position = 1
        elif (short_entry or (position == -1 and not short_exit)) and position >= 0:
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

name = "6h_Donchian20_WeeklyTrend_Pullback_VolumeSpike"
timeframe = "6h"
leverage = 1.0