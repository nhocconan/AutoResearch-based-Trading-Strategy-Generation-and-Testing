#!/usr/bin/env python3
"""
12h_1d_Weekly_Pivot_Breakout_v1
Hypothesis: Use weekly pivot levels with volume confirmation on 12h timeframe.
Long when price breaks above weekly R4 with volume > 1.5x 20-period average,
short when breaks below weekly S4 with volume > 1.5x 20-period average.
Weekly pivots provide strong institutional levels; volume confirms breakout strength.
Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag.
Works in bull via breakouts above resistance, in bear via breakdowns below support.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Weekly_Pivot_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    prev_high = df_1w['high'].iloc[-2] if len(df_1w) >= 2 else df_1w['high'].iloc[-1]
    prev_low = df_1w['low'].iloc[-2] if len(df_1w) >= 2 else df_1w['low'].iloc[-1]
    prev_close = df_1w['close'].iloc[-2] if len(df_1w) >= 2 else df_1w['close'].iloc[-1]
    
    # Calculate weekly pivot levels (standard floor trader pivots)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    
    # Weekly R4 and S4 levels
    weekly_r4 = prev_close + range_val * 1.1 * 2  # R4 = Close + 2.2 * Range
    weekly_s4 = prev_close - range_val * 1.1 * 2  # S4 = Close - 2.2 * Range
    
    # Align weekly levels to 12h timeframe
    weekly_r4_array = np.full(len(df_1w), weekly_r4)
    weekly_s4_array = np.full(len(df_1w), weekly_s4)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4_array)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4_array)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume filter
        long_breakout = close[i] > weekly_r4_aligned[i] and vol_ratio[i] > 1.5
        short_breakout = close[i] < weekly_s4_aligned[i] and vol_ratio[i] > 1.5
        
        # Exit conditions: return to weekly pivot
        weekly_pivot_array = np.full(len(df_1w), pivot)
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_array)
        
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