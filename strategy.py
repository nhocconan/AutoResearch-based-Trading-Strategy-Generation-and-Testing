#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 365:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 1-year high/low calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 365:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 252-day high and low (approx 1 trading year)
    high_252d = np.full(len(high_1d), np.nan)
    low_252d = np.full(len(low_1d), np.nan)
    for i in range(252, len(high_1d)):
        high_252d[i] = np.max(high_1d[i-252:i])
        low_252d[i] = np.min(low_1d[i-252:i])
    
    # Align to daily timeframe
    high_252d_aligned = align_htf_to_ltf(prices, df_1d, high_252d)
    low_252d_aligned = align_htf_to_ltf(prices, df_1d, low_252d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 50-week SMA
    sma_50w = np.full(len(close_1w), np.nan)
    for i in range(50, len(close_1w)):
        sma_50w[i] = np.mean(close_1w[i-50:i])
    
    # Align weekly SMA to daily
    sma_50w_aligned = align_htf_to_ltf(prices, df_1w, sma_50w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(365, n):
        # Skip if data not ready
        if (np.isnan(high_252d_aligned[i]) or 
            np.isnan(low_252d_aligned[i]) or
            np.isnan(sma_50w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Daily trend filter: price above/below weekly SMA
        trend_up = close[i] > sma_50w_aligned[i]
        trend_down = close[i] < sma_50w_aligned[i]
        
        # Breakout conditions: new 252-day high/low
        breakout_up = close[i] > high_252d_aligned[i]
        breakout_down = close[i] < low_252d_aligned[i]
        
        # Entry conditions with trend alignment
        long_entry = breakout_up and trend_up
        short_entry = breakout_down and trend_down
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = position == 1 and (breakout_down or not trend_up)
        exit_short = position == -1 and (breakout_up or not trend_down)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_252d_breakout_trend_filter"
timeframe = "1d"
leverage = 1.0