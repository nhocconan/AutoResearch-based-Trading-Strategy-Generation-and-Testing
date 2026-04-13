#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly high/low for trend bias (using weekly high/low of previous week)
    # We use the weekly range to determine bias: if price > weekly high -> bullish bias, < weekly low -> bearish bias
    weekly_high = np.maximum.accumulate(high_1w)
    weekly_low = np.minimum.accumulate(low_1w)
    
    # Get daily data for entry signals (Camarilla pivots)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (based on previous day's range)
    # Camarilla: 
    # H5 = close + 1.1*(high-low)/2
    # H4 = close + 1.1*(high-low)
    # H3 = close + 1.1*(high-low)/1.5
    # L3 = close - 1.1*(high-low)/1.5
    # L4 = close - 1.1*(high-low)
    # L5 = close - 1.1*(high-low)/2
    
    # Calculate daily ranges
    daily_range = high_1d - low_1d
    # Shift by 1 to use previous day's data for today's levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_range = np.roll(daily_range, 1)
    
    # Handle first day
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_range[0] = high_1d[0] - low_1d[0]
    
    # Calculate Camarilla levels (using previous day's data)
    H3 = prev_close + 1.1 * prev_range / 1.5
    L3 = prev_close - 1.1 * prev_range / 1.5
    H4 = prev_close + 1.1 * prev_range
    L4 = prev_close - 1.1 * prev_range
    
    # Align weekly trend bias to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Align daily Camarilla levels to 6h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(H3_aligned[i]) or
            np.isnan(L3_aligned[i]) or
            np.isnan(H4_aligned[i]) or
            np.isnan(L4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend bias
        # Bullish bias: price above weekly high (new weekly high made)
        # Bearish bias: price below weekly low (new weekly low made)
        weekly_bullish = close[i] > weekly_high_aligned[i]
        weekly_bearish = close[i] < weekly_low_aligned[i]
        
        # Entry logic based on Camarilla levels and weekly bias
        # In bullish weekly bias: look for longs at L3/L4 (support), shorts only on H4 break
        # In bearish weekly bias: look for shorts at H3/H4 (resistance), longs only on L4 break
        
        long_entry = False
        short_entry = False
        
        if weekly_bullish:
            # In bullish week: buy at support (L3/L4), sell at resistance (H3/H4)
            if close[i] <= L3_aligned[i] or close[i] <= L4_aligned[i]:
                long_entry = True
            if close[i] >= H4_aligned[i]:
                short_entry = True  # Only short on strong break above H4 in bullish week
        elif weekly_bearish:
            # In bearish week: sell at resistance (H3/H4), buy at support (L3/L4)
            if close[i] >= H3_aligned[i] or close[i] >= H4_aligned[i]:
                short_entry = True
            if close[i] <= L4_aligned[i]:
                long_entry = True  # Only long on strong break below L4 in bearish week
        else:
            # No clear weekly bias: use mean reversion at extreme levels
            if close[i] <= L3_aligned[i]:
                long_entry = True
            if close[i] >= H3_aligned[i]:
                short_entry = True
        
        # Exit logic: reverse position when opposite signal occurs
        exit_long = position == 1 and short_entry
        exit_short = position == -1 and long_entry
        
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

name = "6h_1w_1d_camarilla_weekly_bias_v1"
timeframe = "6h"
leverage = 1.0