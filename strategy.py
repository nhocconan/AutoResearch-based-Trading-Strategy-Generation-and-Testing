#!/usr/bin/env python3
"""
6h Daily Range Breakout with Volume Spike and 1-week Trend Filter
Breakout above/below previous day's range + volume spike + weekly EMA trend filter
Designed to capture momentum in both bull and bear markets with low trade frequency
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for daily range calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate previous day's range (high - low)
    daily_range = high_1d - low_1d
    prev_high = high_1d  # current day's high (will be shifted)
    prev_low = low_1d    # current day's low (will be shifted)
    
    # Align daily range data to 6h
    daily_range_aligned = align_htf_to_ltf(prices, df_1d, daily_range)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike detection (2x 24-period average - 4 days worth)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(daily_range_aligned[i]) or np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        daily_range_val = daily_range_aligned[i]
        prev_high_val = prev_high_aligned[i]
        prev_low_val = prev_low_aligned[i]
        ema_trend = ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: breakout above previous day's high + volume spike + above weekly EMA
            if (price > prev_high_val and 
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: breakout below previous day's low + volume spike + below weekly EMA
            elif (price < prev_low_val and 
                  volume_spike[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below previous day's low or trend reversal
            if price < prev_low_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above previous day's high or trend reversal
            if price > prev_high_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DailyRangeBreakout_Volume_WeeklyTrend"
timeframe = "6h"
leverage = 1.0