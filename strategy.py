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
    
    # Load daily data once for pivot levels and 12h for trend
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_1d) < 2 or len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points and key levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Calculate 12h EMA for trend direction
    close_12h = df_12h['close'].values
    ema_20_12h = np.full(len(close_12h), np.nan)
    ema_multiplier = 2 / (20 + 1)
    ema_20_12h[19] = np.mean(close_12h[:20])
    for i in range(20, len(close_12h)):
        ema_20_12h[i] = (close_12h[i] - ema_20_12h[i-1]) * ema_multiplier + ema_20_12h[i-1]
    
    # 12h trend: 1 for uptrend, -1 for downtrend
    trend_12h = np.full(len(close_12h), 0)
    for i in range(20, len(close_12h)):
        if close_12h[i] > ema_20_12h[i]:
            trend_12h[i] = 1
        elif close_12h[i] < ema_20_12h[i]:
            trend_12h[i] = -1
    
    # Calculate 12h ATR for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr_12h = np.full(len(tr_12h), np.nan)
    
    for i in range(14, len(tr_12h)):
        atr_12h[i] = np.nanmean(tr_12h[i-13:i+1])
    
    # Create arrays for alignment
    pivot_arr = pivot
    r1_arr = r1
    s1_arr = s1
    r2_arr = r2
    s2_arr = s2
    trend_12h_arr = trend_12h
    atr_12h_arr = atr_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # Get aligned daily and 12h data
        pivot_d = align_htf_to_ltf(prices, df_1d, pivot_arr)[i]
        r1_d = align_htf_to_ltf(prices, df_1d, r1_arr)[i]
        s1_d = align_htf_to_ltf(prices, df_1d, s1_arr)[i]
        r2_d = align_htf_to_ltf(prices, df_1d, r2_arr)[i]
        s2_d = align_htf_to_ltf(prices, df_1d, s2_arr)[i]
        trend_12h_i = align_htf_to_ltf(prices, df_12h, trend_12h_arr)[i]
        atr_12h_i = align_htf_to_ltf(prices, df_12h, atr_12h_arr)[i]
        
        if np.isnan(pivot_d) or np.isnan(r1_d) or np.isnan(s1_d) or np.isnan(r2_d) or np.isnan(s2_d) or np.isnan(trend_12h_i) or np.isnan(atr_12h_i):
            continue
        
        # Volatility filter: only trade when volatility is above average
        if atr_12h_i < np.nanmedian(atr_12h):
            continue
        
        if position == 0:
            # Long: 12h uptrend + price above S1 (support) with rejection
            if trend_12h_i == 1 and close[i] > s1_d and low[i] <= s1_d * 1.001:
                position = 1
                signals[i] = position_size
            # Short: 12h downtrend + price below R1 (resistance) with rejection
            elif trend_12h_i == -1 and close[i] < r1_d and high[i] >= r1_d * 0.999:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: 12h trend turns down OR price reaches R2 (resistance target)
            if trend_12h_i == -1 or close[i] >= r2_d:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: 12h trend turns up OR price reaches S2 (support target)
            if trend_12h_i == 1 or close[i] <= s2_d:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_12hTrend_DailyPivot_Rejection_v1"
timeframe = "4h"
leverage = 1.0