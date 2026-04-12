#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Trend_Filter_v1
Hypothesis: Use daily Camarilla pivot levels with 20-period EMA trend filter on 12h timeframe.
Buy when price breaks above H4 level with EMA > 50-period EMA (bullish trend).
Sell when price breaks below L4 level with EMA < 50-period EMA (bearish trend).
Trades only in direction of higher timeframe trend to avoid counter-trend whipsaws.
Designed for low trade frequency (<150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Trend_Filter_v1"
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
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Camarilla levels (based on previous day)
    range_val = prev_high - prev_low
    h4 = prev_close + 1.1 * range_val * 1.1 / 2
    l4 = prev_close - 1.1 * range_val * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    h4_array = np.full(len(df_1d), h4)
    l4_array = np.full(len(df_1d), l4)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_array)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_array)
    
    # Trend filter: 50-period EMA on 12h timeframe
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_slow = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend condition
        bullish_trend = ema_fast[i] > ema_slow[i]
        bearish_trend = ema_fast[i] < ema_slow[i]
        
        # Breakout conditions with trend filter
        long_breakout = close[i] > h4_aligned[i] and bullish_trend
        short_breakout = close[i] < l4_aligned[i] and bearish_trend
        
        # Exit conditions: return to mid-point (pivot)
        pivot = (prev_high + prev_low + prev_close) / 3
        pivot_array = np.full(len(df_1d), pivot)
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_array)
        
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
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