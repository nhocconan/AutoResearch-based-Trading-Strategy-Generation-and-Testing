#!/usr/bin/env python3
name = "1d_Weekly_Pivot_Breakout_Trend_v2"
timeframe = "1d"
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
    
    # Get weekly data from daily prices
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot: use previous week's OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    # Key levels: S1, R1
    weekly_s1 = weekly_pivot - (weekly_range * 0.382)  # Fibonacci 0.382
    weekly_r1 = weekly_pivot + (weekly_range * 0.382)
    
    # Align weekly levels to daily
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    
    # Daily trend filter: EMA 50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly pivot AND above EMA50 with volume
            if (close[i] > weekly_pivot_aligned[i] and 
                close[i] > ema_50[i] and
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly pivot AND below EMA50 with volume
            elif (close[i] < weekly_pivot_aligned[i] and 
                  close[i] < ema_50[i] and
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below weekly S1 OR below EMA50
            if close[i] < weekly_s1_aligned[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above weekly R1 OR above EMA50
            if close[i] > weekly_r1_aligned[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals