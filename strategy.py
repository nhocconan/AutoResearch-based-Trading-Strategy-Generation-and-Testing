#!/usr/bin/env python3
# 12h_Weekly_Trend_Daily_Range_Breakout
# Hypothesis: Uses 1-week trend filter (price above/below EMA200) and breaks above/below
# daily range (high-low) from previous day with volume confirmation. Designed for 12h timeframe
# to capture multi-day moves in both bull and bear markets by aligning with weekly trend.
# Targets 12-30 trades per year with position size 0.25 to minimize fee drag.

name = "12h_Weekly_Trend_Daily_Range_Breakout"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend direction
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get daily data for range calculation (previous day's high-low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily range from previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    daily_range = prev_high - prev_low
    
    # Calculate breakout levels: previous day's high + 50% of range for long,
    # previous day's low - 50% of range for short
    breakout_long = prev_high + daily_range * 0.5
    breakout_short = prev_low - daily_range * 0.5
    
    # Align breakout levels to 12h timeframe
    breakout_long_aligned = align_htf_to_ltf(prices, df_1d, breakout_long)
    breakout_short_aligned = align_htf_to_ltf(prices, df_1d, breakout_short)
    
    # Calculate volume average for confirmation (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # Warmup for volume MA and weekly EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(breakout_long_aligned[i]) or np.isnan(breakout_short_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above breakout level with volume confirmation and weekly uptrend
            if close[i] > breakout_long_aligned[i] and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below breakout level with volume confirmation and weekly downtrend
            elif close[i] < breakout_short_aligned[i] and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below breakout level or trend turns down
            if close[i] < breakout_long_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above breakout level or trend turns up
            if close[i] > breakout_short_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals