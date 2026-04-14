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
    volume = prices['volume'].values
    
    # Load weekly and daily data once for regime and pivot levels
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly ATR for regime detection (trending vs ranging)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = np.full(len(tr_1w), np.nan)
    
    for i in range(14, len(tr_1w)):
        atr_1w[i] = np.nanmean(tr_1w[i-13:i+1])
    
    # Calculate daily pivot levels (resistance/support)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points and key levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Calculate weekly trend direction using price vs 20-period EMA
    ema_20_1w = np.full(len(close_1w), np.nan)
    ema_multiplier = 2 / (20 + 1)
    ema_20_1w[19] = np.mean(close_1w[:20])
    for i in range(20, len(close_1w)):
        ema_20_1w[i] = (close_1w[i] - ema_20_1w[i-1]) * ema_multiplier + ema_20_1w[i-1]
    
    # Weekly trend: 1 for uptrend, -1 for downtrend, 0 for unclear
    weekly_trend = np.full(len(close_1w), 0)
    for i in range(20, len(close_1w)):
        if close_1w[i] > ema_20_1w[i]:
            weekly_trend[i] = 1
        elif close_1w[i] < ema_20_1w[i]:
            weekly_trend[i] = -1
    
    # Create arrays for alignment
    atr_1w_arr = atr_1w
    weekly_trend_arr = weekly_trend
    r1_arr = r1
    s1_arr = s1
    r2_arr = r2
    s2_arr = s2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # Get aligned weekly data
        atr_w = align_htf_to_ltf(prices, df_1w, atr_1w_arr)[i]
        trend_w = align_htf_to_ltf(prices, df_1w, weekly_trend_arr)[i]
        r1_d = align_htf_to_ltf(prices, df_1d, r1_arr)[i]
        s1_d = align_htf_to_ltf(prices, df_1d, s1_arr)[i]
        r2_d = align_htf_to_ltf(prices, df_1d, r2_arr)[i]
        s2_d = align_htf_to_ltf(prices, df_1d, s2_arr)[i]
        
        if np.isnan(atr_w) or np.isnan(trend_w) or np.isnan(r1_d) or np.isnan(s1_d) or np.isnan(r2_d) or np.isnan(s2_d):
            continue
        
        # Volatility filter: only trade when volatility is above average
        if atr_w < np.nanmedian(atr_1w):
            continue
        
        if position == 0:
            # Long: Weekly uptrend + price above S1 (support) with rejection
            if trend_w == 1 and close[i] > s1_d and low[i] <= s1_d * 1.001:
                position = 1
                signals[i] = position_size
            # Short: Weekly downtrend + price below R1 (resistance) with rejection
            elif trend_w == -1 and close[i] < r1_d and high[i] >= r1_d * 0.999:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Weekly trend turns down OR price reaches R2 (resistance target)
            if trend_w == -1 or close[i] >= r2_d:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Weekly trend turns up OR price reaches S2 (support target)
            if trend_w == 1 or close[i] <= s2_d:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyTrend_Pivot_Rejection_v1"
timeframe = "1d"
leverage = 1.0