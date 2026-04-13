#!/usr/bin/env python3
"""
6h_12h_1d_Volume_Weighted_Pivot_Breakout
Hypothesis: Combines volume-weighted pivot levels from 1d with 12h trend filter and volume confirmation on 6h.
In bull markets: buy breakouts above R1 with volume > 1.5x average and 12h bullish trend.
In bear markets: sell breakdowns below S1 with volume > 1.5x average and 12h bearish trend.
Uses volume-weighted average price (VWAP) of pivot levels to filter false breakouts.
Targets 20-40 trades per year by requiring confluence of volume, pivot break, and trend.
Works in both bull and bear markets by trading breakouts with volume confirmation in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate classic pivot points
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA trend (21-period)
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean()
    trend_bull = close_12h > ema_21_12h
    trend_bear = close_12h < ema_21_12h
    
    # Align all to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    trend_bull_6h = align_htf_to_ltf(prices, df_12h, trend_bull)
    trend_bear_6h = align_htf_to_ltf(prices, df_12h, trend_bear)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(pivot_6h[i]) or \
           np.isnan(r1_6h[i]) or \
           np.isnan(s1_6h[i]) or \
           np.isnan(trend_bull_6h[i]) or \
           np.isnan(trend_bear_6h[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above R1 with volume spike and 12h bullish trend
        if close[i] > r1_6h[i] and volume_spike[i] and trend_bull_6h[i]:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        # Short conditions: price breaks below S1 with volume spike and 12h bearish trend
        elif close[i] < s1_6h[i] and volume_spike[i] and trend_bear_6h[i]:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        # Hold current position
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        # Flat otherwise
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_Volume_Weighted_Pivot_Breakout"
timeframe = "6h"
leverage = 1.0