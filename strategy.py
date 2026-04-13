#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Weekly data for Donchian channels (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily data for 6-hour bars
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Vectorized rolling window with min_periods
    high_roll = pd.Series(high_1w)
    low_roll = pd.Series(low_1w)
    donchian_high = high_roll.rolling(window=20, min_periods=20).max().values
    donchian_low = low_roll.rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to 6-hour timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Daily range for weekly pivot calculation
    # Using previous week's OHLC for current week's pivot (no look-ahead)
    prev_week_high = np.roll(high_1w, 1)
    prev_week_low = np.roll(low_1w, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_high[0] = high_1w[0]
    prev_week_low[0] = low_1w[0]
    prev_week_close[0] = df_1w['close'].values[0]
    
    # Weekly pivot point calculation
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Daily volume confirmation (volume > 1.5x 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_condition = vol_1d > (vol_ma_20 * 1.5)
    vol_condition_aligned = align_htf_to_ltf(prices, df_1d, vol_condition)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_condition_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: long when price > weekly Donchian high, short when price < weekly Donchian low
        long_trend = close[i] > donchian_high_aligned[i]
        short_trend = close[i] < donchian_low_aligned[i]
        
        # Pivot filter: long when price above weekly pivot, short when below
        long_pivot = close[i] > weekly_pivot_aligned[i]
        short_pivot = close[i] < weekly_pivot_aligned[i]
        
        # Volume condition
        vol_ok = vol_condition_aligned[i]
        
        # Entry conditions
        long_entry = long_trend and long_pivot and vol_ok
        short_entry = short_trend and short_pivot and vol_ok
        
        # Exit conditions: reverse when trend changes
        long_exit = not long_trend
        short_exit = not short_trend
        
        if position == 0:
            if long_entry:
                position = 1
                signals[i] = position_size
            elif short_entry:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Donchian_Pivot_Trend_Filter"
timeframe = "6h"
leverage = 1.0