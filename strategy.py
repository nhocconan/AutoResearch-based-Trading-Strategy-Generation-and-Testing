#!/usr/bin/env python3
"""
4h_12h_Camarilla_Breakout_Trend_Filter_v1
Hypothesis: Use 12h timeframe for trend direction (EMA20 vs EMA50) and 1d for daily Camarilla pivot levels.
Enter long when price breaks above H4 level on 4h timeframe with bullish trend on 12h.
Enter short when price breaks below L4 level on 4h timeframe with bearish trend on 12h.
Exit when price returns to daily pivot point.
Uses discrete position sizing (0.30) to minimize churn and keep trade frequency low (target 20-50/year).
Designed to work in both bull and bear markets by following higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Breakout_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Camarilla levels (previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2]
    prev_low = df_1d['low'].iloc[-2]
    prev_close = df_1d['close'].iloc[-2]
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    h4 = prev_close + 1.1 * range_val * 1.1 / 2  # H4 = Close + 1.1*(High-Low)*1.1/2
    l4 = prev_close - 1.1 * range_val * 1.1 / 2  # L4 = Close - 1.1*(High-Low)*1.1/2
    pivot = (prev_high + prev_low + prev_close) / 3  # Daily pivot point
    
    # Create arrays for each day and align to 4h timeframe
    h4_array = np.full(len(df_1d), h4)
    l4_array = np.full(len(df_1d), l4)
    pivot_array = np.full(len(df_1d), pivot)
    
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_array)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_array)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_array)
    
    # 12h trend filter: EMA20 vs EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMAs to 4h timeframe
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Trend condition on 12h
        bullish_trend = ema_20_12h_aligned[i] > ema_50_12h_aligned[i]
        bearish_trend = ema_20_12h_aligned[i] < ema_50_12h_aligned[i]
        
        # Breakout conditions with trend filter
        long_breakout = close[i] > h4_aligned[i] and bullish_trend
        short_breakout = close[i] < l4_aligned[i] and bearish_trend
        
        # Exit conditions: return to daily pivot
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals