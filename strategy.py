#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot filter.
    # Long when price breaks above 6h Donchian(20) high AND 1d weekly pivot shows bullish bias (close > weekly pivot).
    # Short when price breaks below 6h Donchian(20) low AND 1d weekly pivot shows bearish bias (close < weekly pivot).
    # Uses volume confirmation: current 6h volume > 1.5 * 20-period average volume.
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
    # Works in both bull and bear: breakouts capture trends, weekly pivot filter avoids counter-trend trades.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot (using prior week's daily OHLC)
    # For simplicity, use prior 5 trading days (1 week) to calculate weekly pivot
    if len(high_1d) >= 5:
        # Rolling window of 5 days for weekly high/low/close
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    else:
        weekly_pivot = np.full_like(close_1d, np.nan)
    
    # Align HTF weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate Donchian(20) on 6h timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume confirmation
        long_breakout = (close[i] > donchian_high[i]) and volume_confirm[i]
        short_breakout = (close[i] < donchian_low[i]) and volume_confirm[i]
        
        # Weekly pivot filter: only trade in direction of weekly bias
        long_filter = close[i] > weekly_pivot_aligned[i]  # bullish bias
        short_filter = close[i] < weekly_pivot_aligned[i]  # bearish bias
        
        long_entry = long_breakout and long_filter
        short_entry = short_breakout and short_filter
        
        # Exit conditions: opposite Donchian breakout
        long_exit = close[i] < donchian_low[i]
        short_exit = close[i] > donchian_high[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "6h_1d_donchian_breakout_weekly_pivot_filter_v1"
timeframe = "6h"
leverage = 1.0