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
    
    # Get weekly data for pivot calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data for volume and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly pivots: PP = (H+L+C)/3, R1 = 2*PP-L, S1 = 2*PP-H, R2 = PP+(H-L), S2 = PP-(H-L)
    pivot_point = np.full_like(close_1w, np.nan)
    r1 = np.full_like(close_1w, np.nan)
    s1 = np.full_like(close_1w, np.nan)
    r2 = np.full_like(close_1w, np.nan)
    s2 = np.full_like(close_1w, np.nan)
    
    for i in range(len(close_1w)):
        pivot_point[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        r1[i] = 2 * pivot_point[i] - low_1w[i]
        s1[i] = 2 * pivot_point[i] - high_1w[i]
        r2[i] = pivot_point[i] + (high_1w[i] - low_1w[i])
        s2[i] = pivot_point[i] - (high_1w[i] - low_1w[i])
    
    # Calculate 10-period EMA on daily for trend filter
    ema_10 = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 20-period volume average on daily
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 6h timeframe
    pivot_point_6h = align_htf_to_ltf(prices, df_1w, pivot_point)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    ema_10_6h = align_htf_to_ltf(prices, df_1d, ema_10)
    vol_ma_20_6h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA and weekly data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_point_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(ema_10_6h[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_10_6h[i]
        downtrend = close[i] < ema_10_6h[i]
        
        # Volume filter: current volume above average
        volume_filter = volume[i] > vol_ma_20_6h[i]
        
        if position == 0:
            # Long: price breaks above S2 with uptrend and volume confirmation
            if close[i] > s2_6h[i] and uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R2 with downtrend and volume confirmation
            elif close[i] < r2_6h[i] and downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot OR trend reverses
            if (close[i] < pivot_point_6h[i]) or (not uptrend):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot OR trend reverses
            if (close[i] > pivot_point_6h[i]) or (not downtrend):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_S2_S1_Breakout_Volume_EMA10Filter_v1"
timeframe = "6h"
leverage = 1.0