#!/usr/bin/env python3
name = "6h_Donchian_20_WeeklyPivot_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point and R1/S1
    pivot = np.full(len(high_1w), np.nan)
    R1 = np.full(len(high_1w), np.nan)
    S1 = np.full(len(high_1w), np.nan)
    
    for i in range(1, len(high_1w)):
        prev_high = high_1w[i-1]
        prev_low = low_1w[i-1]
        prev_close = close_1w[i-1]
        pivot[i] = (prev_high + prev_low + prev_close) / 3
        R1[i] = 2 * pivot[i] - prev_low
        S1[i] = 2 * pivot[i] - prev_high
    
    # Get daily data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channel (20-day)
    upper = np.full(len(high_1d), np.nan)
    lower = np.full(len(high_1d), np.nan)
    
    for i in range(len(high_1d)):
        if i >= 20:
            upper[i] = np.max(high_1d[i-20:i])
            lower[i] = np.min(low_1d[i-20:i])
        else:
            upper[i] = np.max(high_1d[:i+1])
            lower[i] = np.min(low_1d[:i+1])
    
    # Get 4h data for trend filter (ADX)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX (14)
    def calculate_adx(high, low, close, period=14):
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
        tr[0] = 0
        plus_dm[0] = 0
        minus_dm[0] = 0
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_di = 100 * (np.convolve(plus_dm, np.ones(period)/period, mode='full')[:len(plus_dm)] / atr)
        minus_di = 100 * (np.convolve(minus_dm, np.ones(period)/period, mode='full')[:len(minus_dm)] / atr)
        dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = np.zeros_like(dx)
        adx[period-1] = np.mean(dx[:period]) if period <= len(dx) else 0
        for i in range(period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_trend = adx_4h > 25
    
    # Align indicators to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    adx_trend_aligned = align_htf_to_ltf(prices, df_4h, adx_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or
            np.isnan(adx_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + price > weekly R1 + ADX trend
            if (close[i] > upper_aligned[i] and 
                close[i] > R1_aligned[i] and 
                adx_trend_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + price < weekly S1 + ADX trend
            elif (close[i] < lower_aligned[i] and 
                  close[i] < S1_aligned[i] and 
                  adx_trend_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian or trend weakens
            if (close[i] < lower_aligned[i] or 
                not adx_trend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper Donchian or trend weakens
            if (close[i] > upper_aligned[i] or 
                not adx_trend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals