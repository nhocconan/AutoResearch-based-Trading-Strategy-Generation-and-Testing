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
    
    # Get 1d data for weekly pivot calculation (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from daily data
    # Use last 5 days to approximate weekly OHLC
    weekly_high = np.zeros(len(close_1d))
    weekly_low = np.zeros(len(close_1d))
    weekly_close = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i >= 4:
            weekly_high[i] = np.max(high_1d[i-4:i+1])
            weekly_low[i] = np.min(low_1d[i-4:i+1])
            weekly_close[i] = close_1d[i]
        else:
            weekly_high[i] = np.max(high_1d[:i+1])
            weekly_low[i] = np.min(low_1d[:i+1])
            weekly_close[i] = close_1d[i]
    
    # Weekly pivot calculation
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: current volume above 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1w EMA
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Fade at extreme weekly pivot levels (S3/R3)
        fade_at_s3 = close[i] <= weekly_s3_aligned[i] and volume_filter
        fade_at_r3 = close[i] >= weekly_r3_aligned[i] and volume_filter
        
        # Breakout continuation at stronger levels (S2/R2)
        break_at_s2 = close[i] < weekly_s2_aligned[i] and uptrend and volume_filter
        break_at_r2 = close[i] > weekly_r2_aligned[i] and downtrend and volume_filter
        
        long_entry = fade_at_s3 or break_at_s2
        short_entry = fade_at_r3 or break_at_r2
        
        # Exit conditions: mean reversion to weekly pivot
        if position == 1:
            exit_condition = close[i] >= weekly_pivot_aligned[i]
        elif position == -1:
            exit_condition = close[i] <= weekly_pivot_aligned[i]
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
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

name = "6h_WeeklyPivot_Fade_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0