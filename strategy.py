#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot levels (using Monday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using Monday's data
    # We'll use the first day of each week (Monday) OHLC for weekly pivot
    # For simplicity, we'll use daily OHLC and calculate weekly pivot as:
    # Weekly High = max of last 5 days high
    # Weekly Low = min of last 5 days low
    # Weekly Close = last day's close
    # But since we need Monday's OHLC, we'll approximate with:
    # Weekly Pivot = (Weekly High + Weekly Low + Weekly Close) / 3
    # Using 5-day lookback for weekly high/low
    
    # Calculate 5-day rolling high and low for weekly approximation
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    # For weekly close, we use the current day's close (simplified)
    weekly_close = close_1d
    
    # Weekly pivot points
    weekly_pivot = (high_5d + low_5d + weekly_close) / 3.0
    weekly_range = high_5d - low_5d
    r1_weekly = weekly_pivot + (weekly_range * 1.0)
    s1_weekly = weekly_pivot - (weekly_range * 1.0)
    r2_weekly = weekly_pivot + (weekly_range * 2.0)
    s2_weekly = weekly_pivot - (weekly_range * 2.0)
    
    # Align weekly pivot levels to 6h timeframe
    r1_weekly_aligned = align_htf_to_ltf(prices, df_1d, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_1d, s1_weekly)
    r2_weekly_aligned = align_htf_to_ltf(prices, df_1d, r2_weekly)
    s2_weekly_aligned = align_htf_to_ltf(prices, df_1d, s2_weekly)
    
    # Get 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_weekly_aligned[i]) or np.isnan(s1_weekly_aligned[i]) or 
            np.isnan(r2_weekly_aligned[i]) or np.isnan(s2_weekly_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 weekly with volume and above 1d EMA50
            if close[i] > r1_weekly_aligned[i] and volume_filter[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 weekly with volume and below 1d EMA50
            elif close[i] < s1_weekly_aligned[i] and volume_filter[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S2 weekly
            if close[i] < s2_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R2 weekly
            if close[i] > r2_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1S1_R2S2_Breakout_Volume_EMA50Filter"
timeframe = "6h"
leverage = 1.0