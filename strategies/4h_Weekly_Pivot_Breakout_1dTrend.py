#!/usr/bin/env python3
name = "4h_Weekly_Pivot_Breakout_1dTrend"
timeframe = "4h"
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
    
    # Get 1d data for weekly pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Weekly pivot from 1d data (last 5 days)
    weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    # Weekly support/resistance levels
    weekly_r1 = weekly_pivot + (weekly_range * 1.1 / 12)  # R1
    weekly_s1 = weekly_pivot - (weekly_range * 1.1 / 12)  # S1
    weekly_r2 = weekly_pivot + (weekly_range * 1.1 / 6)   # R2
    weekly_s2 = weekly_pivot - (weekly_range * 1.1 / 6)   # S2
    
    # Align weekly levels to 4h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # 1d trend: EMA 34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r2_aligned[i]) or
            np.isnan(weekly_s2_aligned[i]) or np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly pivot AND above 1d EMA34
            if (close[i] > weekly_pivot_aligned[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly pivot AND below 1d EMA34
            elif (close[i] < weekly_pivot_aligned[i] and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below weekly S1 OR below 1d EMA34
            if close[i] < weekly_s1_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above weekly R1 OR above 1d EMA34
            if close[i] > weekly_r1_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals