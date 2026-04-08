#!/usr/bin/env python3
"""
12h_1w1d_camarilla_volume_trend_v1
Hypothesis: Use weekly and daily trend filters with 12-hour Camarilla pivot levels to capture trend continuation moves with volume confirmation. Designed for low trade frequency (15-35/year) to minimize fee drag while capturing significant market moves in both bull and bear regimes. Weekly trend avoids counter-trend trades, daily trend provides intermediate filter, and 12h Camarilla levels provide precise entry/exit points.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w1d_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    if len(high) < 1:
        return np.full(len(high), np.nan), np.full(len(high), np.nan)
    
    pivot = (high + low + close) / 3.0
    range_val = high - low
    
    H3 = pivot + (range_val * 1.1 / 4)
    L3 = pivot - (range_val * 1.1 / 4)
    
    return H3, L3

def calculate_ema(close, period):
    """Calculate EMA with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    ema = np.full_like(close, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for primary trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for intermediate trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 12-hour data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = calculate_ema(close_1w, 50)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = calculate_ema(close_1d, 50)
    
    # Calculate 12-hour Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_H3_12h, camarilla_L3_12h = calculate_camarilla(high_12h, low_12h, close_12h)
    
    # Align indicators to 12-hour timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    camarilla_H3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_H3_12h)
    camarilla_L3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_L3_12h)
    
    # Volume confirmation: 50-period average
    vol_ma = np.full(n, np.nan)
    for i in range(50, n):
        vol_ma[i] = np.mean(volume[i-50:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(camarilla_H3_12h_aligned[i]) or np.isnan(camarilla_L3_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        weekly_trend_up = price > ema_50_1w_aligned[i]
        daily_trend_up = price > ema_50_1d_aligned[i]
        H3 = camarilla_H3_12h_aligned[i]
        L3 = camarilla_L3_12h_aligned[i]
        
        if position == 1:  # Long
            # Exit: price closes below L3 or trend turns against position
            if price < L3 or not (weekly_trend_up and daily_trend_up):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price closes above H3 or trend turns against position
            if price > H3 or not (not weekly_trend_up and not daily_trend_up):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches/crosses above H3 with volume expansion and both trends up
            if price >= H3 and vol_ratio > 2.0 and weekly_trend_up and daily_trend_up:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches/crosses below L3 with volume expansion and both trends down
            elif price <= L3 and vol_ratio > 2.0 and not weekly_trend_up and not daily_trend_up:
                position = -1
                signals[i] = -0.25
    
    return signals