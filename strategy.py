#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_TrendFilter_v1
Hypothesis: Combines 6h Donchian(20) breakouts with weekly pivot direction (from weekly high/low) and 1d EMA50 trend filter.
Weekly pivot provides structural bias, Donchian captures breakouts, EMA50 filters counter-trend noise.
Designed for low trade frequency (15-30 trades/year) to minimize fee drag while capturing strong trending moves.
Works in bull markets via breakout continuation and in bear markets via breakdown continuation with trend alignment.
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
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot bias: based on weekly range position
    weekly_range = high_1w - low_1w
    weekly_mid = (high_1w + low_1w) / 2.0
    # Bias: 1 if close in upper half, -1 if in lower half, 0 if middle
    weekly_bias = np.where(close_1w > weekly_mid, 1, np.where(close_1w < weekly_mid, -1, 0))
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly bias and EMA50 alignment to 6h
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian(20) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Start after Donchian warmup
    start_idx = lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_bias_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        weekly_bias_val = weekly_bias_aligned[i]
        ema50 = ema50_1d_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        if position == 0:
            # Long conditions: bullish bias, price above EMA50, break above upper channel
            if weekly_bias_val > 0 and close_val > ema50 and close_val > upper_channel:
                signals[i] = size
                position = 1
            # Short conditions: bearish bias, price below EMA50, break below lower channel
            elif weekly_bias_val < 0 and close_val < ema50 and close_val < lower_channel:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price re-enters channel or trend/bias reversal
            if close_val < upper_channel:  # Re-enter channel
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters channel or trend/bias reversal
            if close_val > lower_channel:  # Re-enter channel
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0