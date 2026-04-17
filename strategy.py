#!/usr/bin/env python3
"""
1d_1w_1w_Return_Trend_Filter
Hypothesis: Use 1-week return as a trend filter on daily timeframe. Go long when 1-week return is positive and price breaks above 20-day high with volume confirmation; go short when 1-week return is negative and price breaks below 20-day low with volume confirmation. This strategy aims to capture medium-term momentum while avoiding counter-trend trades, with low trade frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for 20-day high/low and 5-day week return
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 20-day high/low for breakout ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    high_20 = high_series.rolling(window=20, min_periods=20).max().values
    low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 1-week return trend filter (using 1w data) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 5-period return (1 week = 5 trading days)
    ret_5 = np.zeros_like(close_1w)
    ret_5[5:] = (close_1w[5:] - close_1w[:-5]) / close_1w[:-5]
    
    # Align 1-week return to daily timeframe
    ret_5_aligned = align_htf_to_ltf(prices, df_1w, ret_5)
    
    # === 20-day volume average for confirmation ===
    volume_series = pd.Series(volume)
    vol_avg_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 20  # For 20-day indicators
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or
            np.isnan(ret_5_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current day's volume for confirmation
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-day average volume
        vol_filter = vol_current > 1.5 * vol_avg_20[i]
        
        # Trend filter from 1-week return
        trend_up = ret_5_aligned[i] > 0
        trend_down = ret_5_aligned[i] < 0
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above 20-day high + volume filter + up-trend
            if close[i] > high_20[i] and vol_filter and trend_up:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below 20-day low + volume filter + down-trend
            elif close[i] < low_20[i] and vol_filter and trend_down:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse when opposite breakout occurs
        elif position == 1:
            # Exit long when price breaks below 20-day low (trend reversal)
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above 20-day high (trend reversal)
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_1w_Return_Trend_Filter"
timeframe = "1d"
leverage = 1.0