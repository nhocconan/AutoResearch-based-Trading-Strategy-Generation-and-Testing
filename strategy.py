#!/usr/bin/env python3
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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        multiplier = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] - ema_20_1w[i-1]) * multiplier + ema_20_1w[i-1]
    
    # Weekly trend: 1=up, -1=down, 0=neutral
    weekly_trend = np.full(len(close_1w), 0)
    for i in range(20, len(close_1w)):
        if close_1w[i] > ema_20_1w[i]:
            weekly_trend[i] = 1
        elif close_1w[i] < ema_20_1w[i]:
            weekly_trend[i] = -1
    
    # Load daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d  # Resistance 1
    s1 = 2 * pivot - high_1d  # Support 1
    
    # Align HTF data to LTF
    trend_w_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Get aligned values
        trend_w = trend_w_aligned[i]
        r1_d = r1_d_aligned[i]
        s1_d = s1_d_aligned[i]
        
        if np.isnan(trend_w) or np.isnan(r1_d) or np.isnan(s1_d):
            continue
        
        if position == 0:
            # Long: Weekly uptrend + price crosses above S1 (support) with rejection
            if trend_w == 1 and close[i] > s1_d and low[i] <= s1_d * 1.002:
                position = 1
                signals[i] = position_size
            # Short: Weekly downtrend + price crosses below R1 (resistance) with rejection
            elif trend_w == -1 and close[i] < r1_d and high[i] >= r1_d * 0.998:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Weekly trend turns down OR price reaches R1 (resistance target)
            if trend_w == -1 or close[i] >= r1_d:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Weekly trend turns up OR price reaches S1 (support target)
            if trend_w == 1 or close[i] <= s1_d:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_WeeklyTrend_Pivot_Rejection_v2"
timeframe = "12h"
leverage = 1.0