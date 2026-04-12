#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
    # Weekly pivot from 1d provides institutional bias (long above weekly pivot, short below)
    # Donchian breakout captures momentum in direction of weekly pivot
    # Volume confirmation ensures institutional participation
    # Works in bull/bear by aligning with weekly structure
    # Target: 12-37 trades/year per symbol (50-150 over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly pivot (using prior week's OHLC)
    # Approximate week as 5 trading days (skip weekends)
    weekly_pivot = np.full(len(df_1d), np.nan)
    weekly_high = np.full(len(df_1d), np.nan)
    weekly_low = np.full(len(df_1d), np.nan)
    weekly_close = np.full(len(df_1d), np.nan)
    
    # Use 5-day lookback for weekly OHLC (approximation)
    lookback = 5
    for i in range(lookback, len(df_1d)):
        # Prior week's OHLC (5 days ago)
        week_high = np.max(high_1d[i-lookback:i])
        week_low = np.min(low_1d[i-lookback:i])
        week_close = close_1d[i-1]  # yesterday's close
        
        weekly_high[i] = week_high
        weekly_low[i] = week_low
        weekly_close[i] = week_close
        # Weekly pivot = (H + L + C) / 3
        weekly_pivot[i] = (week_high + week_low + week_close) / 3
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # 6h Donchian channels (20-period)
    lookback_period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback_period - 1, n):
        highest_high[i] = np.max(h[i-lookback_period+1:i+1])
        lowest_low[i] = np.min(l[i-lookback_period+1:i+1])
    
    # 1d volume spike filter (current volume > 2.0 * 20-day average)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = volume > 2.0 * vol_ma_20_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine bias from weekly pivot
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        # Breakout conditions with volume confirmation
        long_breakout = close[i] > highest_high[i] and volume_spike[i]
        short_breakout = close[i] < lowest_low[i] and volume_spike[i]
        
        # Exit conditions: reversal or volume dropout
        long_exit = close[i] < lowest_low[i] or (not volume_spike[i])
        short_exit = close[i] > highest_high[i] or (not volume_spike[i])
        
        # Execute trades aligned with weekly bias
        if long_breakout and bullish_bias and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and bearish_bias and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_weekly_pivot_donchian_breakout_v1"
timeframe = "6h"
leverage = 1.0