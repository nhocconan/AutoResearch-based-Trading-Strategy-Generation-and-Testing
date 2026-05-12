#!/usr/bin/env python3
# 6H_DONCHIAN20_WEEKLY_PIVOT_BREAKOUT_VOLUME
# Hypothesis: Donchian(20) breakout on 6h with weekly pivot direction filter.
# In weekly uptrend (price above weekly pivot), go long on Donchian(20) breakout with volume confirmation.
# In weekly downtrend (price below weekly pivot), go short on Donchian(20) breakdown with volume confirmation.
# Weekly pivot acts as trend filter to avoid counter-trend trades, Donchian captures breakouts.
# Volume confirmation reduces false breakouts. Works in both bull and bear markets.
# Target: 15-25 trades/year on 6h timeframe.

name = "6H_DONCHIAN20_WEEKLY_PIVOT_BREAKOUT_VOLUME"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly data for pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly pivot points (using prior week's H, L, C)
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    wp = (wh + wl + wc) / 3.0  # Weekly pivot
    
    # Align weekly pivot to 6h
    wp_aligned = align_htf_to_ltf(prices, df_1w, wp)
    
    # Donchian(20) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian lookback and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(wp_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: weekly uptrend (price above weekly pivot) + Donchian breakout + volume
            if (close[i] > wp_aligned[i] and 
                high[i] > highest_high[i-1] and  # Breakout above prior Donchian high
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: weekly downtrend (price below weekly pivot) + Donchian breakdown + volume
            elif (close[i] < wp_aligned[i] and 
                  low[i] < lowest_low[i-1] and  # Breakdown below prior Donchian low
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or weekly trend fails
            if (low[i] < lowest_low[i-1] or 
                close[i] < wp_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or weekly trend fails
            if (high[i] > highest_high[i-1] or 
                close[i] > wp_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals