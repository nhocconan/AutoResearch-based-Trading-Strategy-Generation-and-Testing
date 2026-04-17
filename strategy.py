#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_VolumeFilter
Hypothesis: Donchian(20) breakouts in the direction of the weekly pivot trend (price above/below weekly pivot) with volume confirmation capture strong trends while avoiding false breakouts in chop. Weekly pivot provides structural bias; volume ensures momentum. Designed for 6B timeframe to reduce whipsaw and work in both bull (breakout continuation) and bear (avoid false longs via pivot filter).
"""

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
    
    # Donchian(20) channels
    def donchian_channels(high, low, window):
        dc_high = np.full_like(high, np.nan)
        dc_low = np.full_like(low, np.nan)
        for i in range(window-1, len(high)):
            dc_high[i] = np.max(high[i-window+1:i+1])
            dc_low[i] = np.min(low[i-window+1:i+1])
        return dc_high, dc_low
    
    dc_high, dc_low = donchian_channels(high, low, 20)
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot to 6t timeframe (with 1-bar delay for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Volume confirmation (20-period average on 6h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20)  # Donchian(20), volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dc_high[i]) or 
            np.isnan(dc_low[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Breakout conditions
        bullish_breakout = close[i] > dc_high[i-1]  # break above prior period high
        bearish_breakout = close[i] < dc_low[i-1]   # break below prior period low
        
        # Weekly pivot bias: price above pivot = bullish bias, below = bearish bias
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: bullish breakout + bullish bias + volume filter
            if bullish_breakout and bullish_bias and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + bearish bias + volume filter
            elif bearish_breakout and bearish_bias and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout or price below weekly pivot
            if bearish_breakout or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout or price above weekly pivot
            if bullish_breakout or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_VolumeFilter"
timeframe = "6h"
leverage = 1.0