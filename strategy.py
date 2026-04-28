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
    
    # Get daily data for weekly pivot calculation (using Monday of week)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data
    # Group by week: Monday to Friday (assuming 5 trading days per week)
    # We'll use the prior week's high, low, close for current week pivot
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Calculate weekly high/low/close using 5-day rolling window (prior week)
    # Using minimum 3 days to avoid NaN issues
    high_wk = pd.Series(high_d).rolling(window=5, min_periods=3).max().shift(1).values
    low_wk = pd.Series(low_d).rolling(window=5, min_periods=3).min().shift(1).values
    close_wk = pd.Series(close_d).rolling(window=5, min_periods=3).last().shift(1).values
    
    # Weekly pivot point
    pivot_wk = (high_wk + low_wk + close_wk) / 3
    range_wk = high_wk - low_wk
    
    # Weekly support/resistance levels (using pivot formula)
    r1_wk = pivot_wk + (high_wk - low_wk)
    s1_wk = pivot_wk - (high_wk - low_wk)
    r2_wk = pivot_wk + 2 * (high_wk - low_wk)
    s2_wk = pivot_wk - 2 * (high_wk - low_wk)
    
    # Align to 6h timeframe
    r2_wk_aligned = align_htf_to_ltf(prices, df_1d, r2_wk)
    s2_wk_aligned = align_htf_to_ltf(prices, df_1d, s2_wk)
    r1_wk_aligned = align_htf_to_ltf(prices, df_1d, r1_wk)
    s1_wk_aligned = align_htf_to_ltf(prices, df_1d, s1_wk)
    
    # Get weekly data for trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 0-24 UTC (trade all hours for 6h timeframe)
    # No session filter for 6h to capture global movements
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_wk_aligned[i]) or np.isnan(s2_wk_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below weekly EMA20
        trend_up = close[i] > ema20_1w_aligned[i]
        trend_down = close[i] < ema20_1w_aligned[i]
        
        # Entry conditions: 
        # Long: break above weekly S2 with upward trend and volume
        # Short: break below weekly R2 with downward trend and volume
        long_breakout = close[i] > s2_wk_aligned[i]
        short_breakout = close[i] < r2_wk_aligned[i]
        
        long_entry = long_breakout and vol_filter and trend_up
        short_entry = short_breakout and vol_filter and trend_down
        
        # Exit conditions: opposite R1/S1 level touch
        long_exit = (close[i] < s1_wk_aligned[i]) and position == 1
        short_exit = (close[i] > r1_wk_aligned[i]) and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_S2R2_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0