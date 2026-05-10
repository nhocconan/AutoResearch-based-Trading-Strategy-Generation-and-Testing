#!/usr/bin/env python3
# 6h_PivotReversal_MultiTimeframe_VolumeFilter
# Hypothesis: Uses 12h pivot points (classic) for mean reversion in ranging markets.
# Long when price touches S1 with bullish divergence (RSI < 30) and volume confirmation.
# Short when price touches R1 with bearish divergence (RSI > 70) and volume confirmation.
# Uses 1d ADX < 25 to filter for ranging markets only.
# Designed for 15-30 trades/year to avoid overtrading and work in both bull and bear markets.

name = "6h_PivotReversal_MultiTimeframe_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot points
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate classic pivot points from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Classic pivot: P = (H + L + C)/3
    # Support 1: S1 = 2*P - H
    # Resistance 1: R1 = 2*P - L
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    s1_12h = 2 * pivot_12h - high_12h
    r1_12h = 2 * pivot_12h - low_12h
    
    # Align pivot levels to 6h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    
    # Get 1d data for ADX ranging filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) for ranging market detection
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smoothed_avg(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Subsequent values are Wilder's smoothing
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = smoothed_avg(tr, 14)
    plus_di = 100 * smoothed_avg(dm_plus, 14) / atr
    minus_di = 100 * smoothed_avg(dm_minus, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smoothed_avg(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate RSI(14) on 6h for overbought/oversold
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    # Wilder's smoothing for RSI
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.full(14, np.nan)], rsi])  # align with price index
    
    # Volume average (24 periods = 4 days of 6h)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in ranging markets (ADX < 25)
        if adx_aligned[i] >= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price at S1 with oversold RSI and volume confirmation
            if (close[i] <= s1_12h_aligned[i] * 1.001 and  # Allow small tolerance
                rsi[i] < 30 and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price at R1 with overbought RSI and volume confirmation
            elif (close[i] >= r1_12h_aligned[i] * 0.999 and  # Allow small tolerance
                  rsi[i] > 70 and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to pivot or RSI overbought
            if (close[i] >= pivot_12h_aligned[i] * 0.999 or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to pivot or RSI oversold
            if (close[i] <= pivot_12h_aligned[i] * 1.001 or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals