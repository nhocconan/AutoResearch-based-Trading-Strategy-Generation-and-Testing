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
    
    # Get 1d data for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week
    # Need daily high/low/close from 5 days ago to 1 day ago
    pivots_high = np.full(len(close_1d), np.nan)
    pivots_low = np.full(len(close_1d), np.nan)
    for i in range(5, len(close_1d)):
        # Use data from i-5 to i-1 (prior 5 trading days = 1 week)
        week_high = np.max(high_1d[i-5:i])
        week_low = np.min(low_1d[i-5:i])
        week_close = close_1d[i-1]  # Previous day close
        pivot = (week_high + week_low + week_close) / 3.0
        r1 = 2 * pivot - week_low
        s1 = 2 * pivot - week_high
        r2 = pivot + (week_high - week_low)
        s2 = pivot - (week_high - week_low)
        r3 = r2 + (week_high - week_low)
        s3 = s2 - (week_high - week_low)
        pivots_high[i] = r3
        pivots_low[i] = s3
    
    # Align weekly pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, pivots_high)
    s3_aligned = align_htf_to_ltf(prices, df_1d, pivots_low)
    
    # Calculate 60-period EMA on 1d (trend filter)
    close_1d_series = pd.Series(close_1d)
    ema_60_1d = close_1d_series.ewm(span=60, adjust=False, min_periods=60).mean().values
    ema_60_aligned = align_htf_to_ltf(prices, df_1d, ema_60_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_60_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA60
        above_ema = close[i] > ema_60_aligned[i]
        below_ema = close[i] < ema_60_aligned[i]
        
        # Entry conditions: touch weekly S3/R3 with trend alignment
        touch_s3 = low[i] <= s3_aligned[i] * 1.001  # Allow small buffer
        touch_r3 = high[i] >= r3_aligned[i] * 0.999
        
        long_entry = touch_s3 and above_ema
        short_entry = touch_r3 and below_ema
        
        # Exit conditions: opposite touch or trend reversal
        exit_long = position == 1 and (touch_r3 or below_ema)
        exit_short = position == -1 and (touch_s3 or above_ema)
        
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

name = "6h_1d_weekly_pivot_touch"
timeframe = "6h"
leverage = 1.0