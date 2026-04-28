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
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot levels from previous week's OHLC
    # Get weekly OHLC from daily data
    weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().shift(1)
    weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().shift(1)
    weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().shift(1)
    
    # Weekly pivot point (P) = (H + L + C) / 3
    weekly_p = (weekly_high + weekly_low + weekly_close) / 3
    
    # Support and resistance levels
    r1 = 2 * weekly_p - weekly_low
    s1 = 2 * weekly_p - weekly_high
    r2 = weekly_p + (weekly_high - weekly_low)
    s2 = weekly_p - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (weekly_p - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_p)
    
    # Align pivot levels to 6h timeframe
    weekly_p_aligned = align_htf_to_ltf(prices, df_1d, weekly_p.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Get weekly data for trend filter (price vs weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly close for trend filter
    weekly_close_series = pd.Series(df_1w['close'].values)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close_series.values)
    
    # Volume filter: above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_p_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(weekly_close_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below weekly close
        trend_up = close[i] > weekly_close_aligned[i]
        trend_down = close[i] < weekly_close_aligned[i]
        
        # Entry conditions: 
        # Long: break above R2 with upward trend and volume
        # Short: break below S2 with downward trend and volume
        long_breakout = close[i] > r2_aligned[i]
        short_breakout = close[i] < s2_aligned[i]
        
        long_entry = long_breakout and vol_filter and trend_up
        short_entry = short_breakout and vol_filter and trend_down
        
        # Exit conditions: 
        # Long exit: price falls below R1 (taking profit at first resistance)
        # Short exit: price rises above S1 (taking profit at first support)
        long_exit = (close[i] < r1_aligned[i]) and position == 1
        short_exit = (close[i] > s1_aligned[i]) and position == -1
        
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

name = "6h_WeeklyPivot_R2S2_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0