#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and key levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Calculate daily trend using 20-period EMA
    ema_20_1d = np.full(len(close_1d), np.nan)
    ema_multiplier = 2 / (20 + 1)
    if len(close_1d) >= 20:
        ema_20_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_20_1d[i] = (close_1d[i] - ema_20_1d[i-1]) * ema_multiplier + ema_20_1d[i-1]
    
    # Daily trend: 1 for uptrend, -1 for downtrend
    daily_trend = np.full(len(close_1d), 0)
    for i in range(20, len(close_1d)):
        if close_1d[i] > ema_20_1d[i]:
            daily_trend[i] = 1
        elif close_1d[i] < ema_20_1d[i]:
            daily_trend[i] = -1
    
    # Align daily data to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned daily data
        piv = pivot_aligned[i]
        r1d = r1_aligned[i]
        s1d = s1_aligned[i]
        trend_d = daily_trend_aligned[i]
        
        if np.isnan(piv) or np.isnan(r1d) or np.isnan(s1d) or np.isnan(trend_d):
            continue
        
        if position == 0:
            # Long: Daily uptrend + price above S1 support with rejection
            if trend_d == 1 and close[i] > s1d and low[i] <= s1d * 1.002:
                position = 1
                signals[i] = position_size
            # Short: Daily downtrend + price below R1 resistance with rejection
            elif trend_d == -1 and close[i] < r1d and high[i] >= r1d * 0.998:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Daily trend turns down OR price reaches R1 resistance
            if trend_d == -1 or close[i] >= r1d:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Daily trend turns up OR price reaches S1 support
            if trend_d == 1 or close[i] <= s1d:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_DailyTrend_Pivot_Rejection_v1"
timeframe = "4h"
leverage = 1.0