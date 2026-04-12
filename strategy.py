#!/usr/bin/env python3
"""
12h_1w_1d_Range_Breakout_With_Trend_Filter
Hypothesis: On 12h timeframe, use weekly range (weekly high-low) to identify expansion/contraction regimes.
When weekly range expands (above 20-period SMA of weekly ranges), breakout trades in direction of 1d trend.
Enter long when price breaks above weekly high with 1d uptrend and volume > 1.5x average.
Enter short when price breaks below weekly low with 1d downtrend and volume > 1.5x average.
Exit when price returns to weekly midpoint or trend reverses.
Designed to capture volatility expansion moves in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Range_Breakout_With_Trend_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR RANGE AND LEVELS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly range and its SMA for volatility regime filter
    weekly_range = high_1w - low_1w
    range_sma = pd.Series(weekly_range).rolling(window=20, min_periods=20).mean().values
    range_expansion = weekly_range > range_sma  # Volatility expansion regime
    
    # Weekly high/low for breakout levels
    weekly_high = high_1w
    weekly_low = low_1w
    weekly_mid = (high_1w + low_1w) / 2.0
    
    # Align weekly data to 12h timeframe
    range_expansion_12h = align_htf_to_ltf(prices, df_1w, range_expansion)
    weekly_high_12h = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_12h = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_mid_12h = align_htf_to_ltf(prices, df_1w, weekly_mid)
    
    # === DAILY TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    sma20_1d_12h = align_htf_to_ltf(prices, df_1d, sma20_1d)
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(range_expansion_12h[i]) or np.isnan(weekly_high_12h[i]) or 
            np.isnan(weekly_low_12h[i]) or np.isnan(sma20_1d_12h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get daily trend
        close_1d_arr = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_arr)
        trend_up = close_1d_aligned[i] > sma20_1d_12h[i]
        trend_down = close_1d_aligned[i] < sma20_1d_12h[i]
        
        # Entry conditions: volatility expansion + breakout + trend + volume
        long_signal = (range_expansion_12h[i] and 
                      close[i] > weekly_high_12h[i] and 
                      trend_up and 
                      vol_ratio[i] > 1.5)
        
        short_signal = (range_expansion_12h[i] and 
                       close[i] < weekly_low_12h[i] and 
                       trend_down and 
                       vol_ratio[i] > 1.5)
        
        # Exit conditions: return to midpoint or trend reversal
        exit_long = (position == 1 and 
                    (close[i] <= weekly_mid_12h[i] or not trend_up))
        exit_short = (position == -1 and 
                     (close[i] >= weekly_mid_12h[i] or not trend_down))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals