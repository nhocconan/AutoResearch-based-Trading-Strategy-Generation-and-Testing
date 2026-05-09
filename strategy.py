#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Russell2000_Strength_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Russell 2000-like strength (high-low ratio)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Russell 2000 strength: weekly high-low ratio > 0.6 (strong trending week)
    weekly_range = df_1w['high'].values - df_1w['low'].values
    weekly_body = abs(df_1w['close'].values - df_1w['open'].values)
    strength_ratio = weekly_body / weekly_range
    strength_ratio = np.where(weekly_range == 0, 0, strength_ratio)
    strong_week = strength_ratio > 0.6
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 6h Donchian breakout (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align all to 6h
    strong_week_6h = align_htf_to_ltf(prices, df_1w, strong_week)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    highest_high_6h = align_htf_to_ltf(prices, df_1w, highest_high)  # Align weekly high
    lowest_low_6h = align_htf_to_ltf(prices, df_1w, lowest_low)      # Align weekly low
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, lookback)  # Need enough data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(strong_week_6h[i]) or np.isnan(ema50_1d_6h[i]) or
            np.isnan(highest_high_6h[i]) or np.isnan(lowest_low_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        strong = strong_week_6h[i]
        trend = ema50_1d_6h[i]
        upper = highest_high_6h[i]
        lower = lowest_low_6h[i]
        
        if position == 0:
            # Enter long: strong weekly trend + price breaks above weekly high + above daily trend
            if strong and close[i] > upper and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: strong weekly trend + price breaks below weekly low + below daily trend
            elif strong and close[i] < lower and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly low (trend failure)
            if close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly high (trend failure)
            if close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals