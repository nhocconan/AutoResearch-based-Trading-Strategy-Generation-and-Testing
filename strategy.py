#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Breakout_v1
Hypothesis: Use daily and 4h Camarilla pivot levels with volume confirmation.
Long when price breaks above H4 (daily) with volume > 1.5x average, short when breaks below L4 (daily).
Use 4h trend filter (EMA20 > EMA50) to avoid counter-trend trades.
Designed for low trade frequency (target: 60-150 total over 4 years) to minimize fee drag.
Works in bull via breakouts, in bear via trend-filtered short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Calculate daily Camarilla levels
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    daily_h4 = prev_close + 1.1 * range_val * 1.1 / 2
    daily_l4 = prev_close - 1.1 * range_val * 1.1 / 2
    
    # Align daily levels to 1h timeframe
    daily_h4_array = np.full(len(df_1d), daily_h4)
    daily_l4_array = np.full(len(df_1d), daily_l4)
    daily_h4_aligned = align_htf_to_ltf(prices, df_1d, daily_h4_array)
    daily_l4_aligned = align_htf_to_ltf(prices, df_1d, daily_l4_array)
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any data invalid
        if (np.isnan(daily_h4_aligned[i]) or np.isnan(daily_l4_aligned[i]) or
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend condition from 4h
        bullish_trend = ema_20_4h_aligned[i] > ema_50_4h_aligned[i]
        bearish_trend = ema_20_4h_aligned[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions with filters
        long_breakout = close[i] > daily_h4_aligned[i] and bullish_trend and vol_spike
        short_breakout = close[i] < daily_l4_aligned[i] and bearish_trend and vol_spike
        
        # Exit conditions: return to midpoint
        daily_pivot = (prev_high + prev_low + prev_close) / 3
        daily_pivot_array = np.full(len(df_1d), daily_pivot)
        daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot_array)
        
        long_exit = close[i] < daily_pivot_aligned[i]
        short_exit = close[i] > daily_pivot_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals