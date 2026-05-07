#!/usr/bin/env python3
"""
1d_Structure_Breakout_With_Volume_and_Trend_Filter
Hypothesis: Daily breakouts of key structural levels (prior day high/low) with volume confirmation and 200-day EMA trend filter. Designed for low trade frequency (7-25/year) to avoid fee drag. Works in both bull and bear markets by aligning with long-term trend direction. Uses discrete position sizing (0.30) to minimize churn.
"""

name = "1d_Structure_Breakout_With_Volume_and_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get daily data for structure levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's high and low (structure levels)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align structure levels to daily timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Volume confirmation: 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Warmup for weekly EMA200
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend using aligned close
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
        if np.isnan(weekly_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        weekly_trend_up = weekly_close_aligned[i] > ema_200_1w_aligned[i]
        weekly_trend_down = weekly_close_aligned[i] < ema_200_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above prior day's high, volume spike, weekly trend up
            if (high[i] > prev_high_aligned[i] and 
                vol_ratio[i] > 2.0 and 
                weekly_trend_up):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below prior day's low, volume spike, weekly trend down
            elif (low[i] < prev_low_aligned[i] and 
                  vol_ratio[i] > 2.0 and 
                  weekly_trend_down):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below prior day's low or weekly trend changes
            if low[i] < prev_low_aligned[i] or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above prior day's high or weekly trend changes
            if high[i] > prev_high_aligned[i] or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals