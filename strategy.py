#!/usr/bin/env python3
# [24888] 12h_1w1d_ema_volume_v1
# Hypothesis: 12-hour EMA trend with weekly EMA filter and volume confirmation.
# Long when price > 12h EMA21 and weekly EMA21 slope > 0 and volume > 1.5x average.
# Short when price < 12h EMA21 and weekly EMA21 slope < 0 and volume > 1.5x average.
# Exit when price crosses back below/above 12h EMA21.
# Designed to work in both bull and bear markets by using weekly trend filter to avoid counter-trend trades.
# Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w1d_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA21
    close_1w = df_1w['close'].values
    ema_21_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 21:
        alpha = 2.0 / (21 + 1)
        ema_21_1w[20] = np.mean(close_1w[:21])
        for i in range(21, len(close_1w)):
            ema_21_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_21_1w[i-1]
    
    # Calculate weekly EMA21 slope (trend direction)
    ema_slope_1w = np.full_like(close_1w, np.nan, dtype=float)
    for i in range(22, len(close_1w)):
        if not np.isnan(ema_21_1w[i]) and not np.isnan(ema_21_1w[i-1]):
            ema_slope_1w[i] = ema_21_1w[i] - ema_21_1w[i-1]
    
    # Get daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(20, len(volume_1d)):
        vol_avg_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Calculate 12-hour EMA21
    ema_21_12h = np.full(n, np.nan)
    if n >= 21:
        alpha = 2.0 / (21 + 1)
        ema_21_12h[20] = np.mean(close[:21])
        for i in range(21, n):
            ema_21_12h[i] = alpha * close[i] + (1 - alpha) * ema_21_12h[i-1]
    
    # Align weekly EMA21 and slope to 12-hour timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    ema_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_1w)
    
    # Align daily volume average to 12-hour timeframe
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_21_12h[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(ema_slope_1w_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_avg_1d_aligned[i] if vol_avg_1d_aligned[i] > 0 else 0
        price = close[i]
        ema12 = ema_21_12h[i]
        weekly_ema = ema_21_1w_aligned[i]
        weekly_slope = ema_slope_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 12h EMA21
            if price < ema12:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 12h EMA21
            if price > ema12:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price > 12h EMA21, weekly EMA up, volume expansion
            if price > ema12 and weekly_slope > 0 and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: price < 12h EMA21, weekly EMA down, volume expansion
            elif price < ema12 and weekly_slope < 0 and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals