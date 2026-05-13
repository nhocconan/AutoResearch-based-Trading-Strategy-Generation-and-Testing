#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_Trend_Filter
Hypothesis: Use weekly pivot points (R2/S2) as dynamic support/resistance. 
Breakout above R2 with 1d EMA50 uptrend and volume confirmation signals long. 
Breakdown below S2 with 1d EMA50 downtrend and volume confirmation signals short.
Weekly pivots provide robust S/R that works in both bull/bear markets, while 
volume confirmation filters false breakouts. Target: 15-30 trades/year per symbol.
"""

name = "6h_WeeklyPivot_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly pivot points (R2/S2)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R2 = P + (H-L), S2 = P - (H-L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    r2 = weekly_pivot + weekly_range
    s2 = weekly_pivot - weekly_range
    
    # Align weekly R2/S2 to 6h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: break above R2, 1d uptrend, volume confirmation
            if close[i] > r2_aligned[i] and uptrend_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S2, 1d downtrend, volume confirmation
            elif close[i] < s2_aligned[i] and downtrend_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below weekly pivot or trend reverses
            weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
            if close[i] < weekly_pivot_aligned[i] or not uptrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above weekly pivot or trend reverses
            weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
            if close[i] > weekly_pivot_aligned[i] or not downtrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals