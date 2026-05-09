#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_HTF_RangeBreakout_With_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for range and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily range (high-low) average
    daily_range = df_1d['high'].values - df_1d['low'].values
    avg_daily_range = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    
    # Daily volume average
    daily_vol = df_1d['volume'].values
    avg_daily_vol = pd.Series(daily_vol).rolling(window=20, min_periods=20).mean().values
    
    # Daily close for trend context
    daily_close = df_1d['close'].values
    ema20_daily = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all to 4h
    avg_range_4h = align_htf_to_ltf(prices, df_1d, avg_daily_range)
    avg_vol_4h = align_htf_to_ltf(prices, df_1d, avg_daily_vol)
    ema20_4h = align_htf_to_ltf(prices, df_1d, ema20_daily)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(avg_range_4h[i]) or np.isnan(avg_vol_4h[i]) or np.isnan(ema20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current day's range reference (from previous day's close)
        day_open_idx = i - (i % 16)  # Start of current day in 4h bars
        if day_open_idx < 0:
            continue
            
        # Get the daily reference values for this day
        day_idx = day_open_idx // 16
        if day_idx >= len(df_1d):
            continue
            
        # Use previous day's close as anchor for range
        prev_close = daily_close[day_idx - 1] if day_idx > 0 else daily_close[0]
        prev_range = avg_daily_range[day_idx - 1] if day_idx > 0 else avg_daily_range[0]
        prev_vol_avg = avg_daily_vol[day_idx - 1] if day_idx > 0 else avg_daily_vol[0]
        
        # Calculate bands based on previous day
        upper_band = prev_close + 0.5 * prev_range
        lower_band = prev_close - 0.5 * prev_range
        vol_threshold = prev_vol_avg * 1.8
        
        if position == 0:
            # Long: break above upper band with volume surge
            if close[i] > upper_band and volume[i] > vol_threshold:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume surge
            elif close[i] < lower_band and volume[i] > vol_threshold:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to mid-point or volume drops
            mid_point = prev_close
            if close[i] < mid_point or volume[i] < vol_threshold * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to mid-point or volume drops
            mid_point = prev_close
            if close[i] > mid_point or volume[i] < vol_threshold * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals