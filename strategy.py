#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Weekly_Pivot_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 8:
        return np.zeros(n)
    
    # Calculate weekly pivot from previous week's OHLC
    # We need to aggregate daily data into weekly periods
    # For simplicity, we'll use the last 7 days as a proxy for the previous week
    if len(df_1d) >= 7:
        # Get the week before the last 7 days (i.e., days -14 to -7)
        week_high = df_1d['high'].iloc[-14:-7].max() if len(df_1d) >= 14 else df_1d['high'].iloc[:-7].max()
        week_low = df_1d['low'].iloc[-14:-7].min() if len(df_1d) >= 14 else df_1d['low'].iloc[:-7].min()
        week_close = df_1d['close'].iloc[-7]  # Close of the day before the last 7 days
    else:
        # Fallback: use available data
        week_high = df_1d['high'].iloc[:-1].max() if len(df_1d) > 1 else df_1d['high'].iloc[-1]
        week_low = df_1d['low'].iloc[:-1].min() if len(df_1d) > 1 else df_1d['low'].iloc[-1]
        week_close = df_1d['close'].iloc[-2] if len(df_1d) > 1 else df_1d['close'].iloc[-1]
    
    # Calculate weekly pivot levels (standard floor trader pivots)
    pivot = (week_high + week_low + week_close) / 3
    range_val = week_high - week_low
    if range_val <= 0:
        return np.zeros(n)
    
    # Weekly R4 and S4 levels
    weekly_r4 = week_close + range_val * 1.1 * 2  # R4 = Close + 2.2 * Range
    weekly_s4 = week_close - range_val * 1.1 * 2  # S4 = Close - 2.2 * Range
    
    # Create arrays for the week and align to 4h
    weekly_r4_array = np.full(len(df_1d), weekly_r4)
    weekly_s4_array = np.full(len(df_1d), weekly_s4)
    weekly_pivot_array = np.full(len(df_1d), pivot)
    
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4_array)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4_array)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_array)
    
    # Volume confirmation: current volume > 1.3x 50-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=50, min_periods=50).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any data invalid
        if (np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume filter
        long_breakout = close[i] > weekly_r4_aligned[i] and vol_ratio[i] > 1.3
        short_breakout = close[i] < weekly_s4_aligned[i] and vol_ratio[i] > 1.3
        
        # Exit conditions: return to weekly pivot
        long_exit = close[i] < weekly_pivot_aligned[i]
        short_exit = close[i] > weekly_pivot_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals