#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation (need previous week's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly high/low/close for pivot points
    # We'll use the previous week's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly pivot: use previous week's OHLC
    # For each day, we need the week that ended on the previous Friday
    weekly_high = np.full(len(close_1d), np.nan)
    weekly_low = np.full(len(close_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    # Simple approach: use 5-day rolling window for weekly data
    # This approximates weekly OHLC from daily data
    for i in range(5, len(close_1d)):
        # Previous 5 days (excluding current day) = previous week
        weekly_high[i] = np.max(high_1d[i-5:i])
        weekly_low[i] = np.min(low_1d[i-5:i])
        weekly_close[i] = close_1d[i-1]  # Previous day's close as weekly close approximation
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_pivot = np.full(len(close_1d), np.nan)
    weekly_r1 = np.full(len(close_1d), np.nan)
    weekly_s1 = np.full(len(close_1d), np.nan)
    weekly_r2 = np.full(len(close_1d), np.nan)
    weekly_s2 = np.full(len(close_1d), np.nan)
    weekly_r3 = np.full(len(close_1d), np.nan)
    weekly_s3 = np.full(len(close_1d), np.nan)
    
    for i in range(5, len(close_1d)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            H = weekly_high[i]
            L = weekly_low[i]
            C = weekly_close[i]
            P = (H + L + C) / 3
            weekly_pivot[i] = P
            weekly_r1[i] = 2 * P - L
            weekly_s1[i] = 2 * P - H
            weekly_r2[i] = P + (H - L)
            weekly_s2[i] = P - (H - L)
            weekly_r3[i] = H + 2 * (P - L)
            weekly_s3[i] = L - 2 * (H - P)
    
    # Get weekly data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20
    ema_period = 20
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: current volume > 1.5x 50-period average (longer MA for 6h)
    vol_ma = np.full(n, np.nan)
    vol_period = 50
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivot, EMA, and volume MA
    start_idx = max(5, ema_period, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(weekly_s3_aligned[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price crosses above weekly R3 + volume spike + above weekly EMA20
            if (price > weekly_r3_aligned[i] and 
                vol_ratio > 1.5 and 
                price > ema_1w_aligned[i]):
                signals[i] = size
                position = 1
            # Short: Price crosses below weekly S3 + volume spike + below weekly EMA20
            elif (price < weekly_s3_aligned[i] and 
                  vol_ratio > 1.5 and 
                  price < ema_1w_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below weekly S3 OR loses weekly trend
            if (price < weekly_s3_aligned[i] or 
                price < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above weekly R3 OR loses weekly trend
            if (price > weekly_r3_aligned[i] or 
                price > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_R3S3_EMA20_Volume"
timeframe = "6h"
leverage = 1.0