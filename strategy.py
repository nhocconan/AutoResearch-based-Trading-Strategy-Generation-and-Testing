#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_TurtleSoup_Reverse_1wTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for weekly trend
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    
    # Calculate 40-period EMA on weekly close
    weekly_ema = pd.Series(weekly_close).ewm(span=40, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Calculate 20-period Donchian channels on 12h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_uptrend = close[i] > weekly_ema_aligned[i]
        weekly_downtrend = close[i] < weekly_ema_aligned[i]
        
        if position == 0:
            # Turtle Soup reversal: false breakout of 20-period Donchian
            # Long setup: price breaks below 20-period low then reverses above it
            if weekly_uptrend and low[i] < low_20[i] and close[i] > low_20[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: price breaks above 20-period high then reverses below it
            elif weekly_downtrend and high[i] > high_20[i] and close[i] < high_20[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches 20-period high or weekly trend changes
            if close[i] >= high_20[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches 20-period low or weekly trend changes
            if close[i] <= low_20[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals