#!/usr/bin/env python3
"""
1d Weekly Pivot Breakout with Volume Confirmation and Trend Filter
Hypothesis: Weekly pivot levels (S1/S2/R1/R2) act as strong support/resistance on daily charts.
Breakouts with volume confirmation indicate institutional participation.
Trend filter (50 EMA) ensures we trade in direction of higher timeframe trend.
Works in both bull and bear markets by following breakout direction.
Target: 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(arr, period):
    """Calculate Exponential Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    ema = np.zeros_like(arr)
    multiplier = 2 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = (arr[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    return ema

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Support and resistance levels
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + range_1w
    s2 = pivot - range_1w
    
    # Align to daily timeframe (use previous week's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume confirmation: current volume > 1.8x 20-day average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.8)
    
    # Trend filter: 50-day EMA
    ema_50 = calculate_ema(close, 50)
    uptrend = close > ema_50
    downtrend = close < ema_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume in uptrend
            if (close[i] > r1_aligned[i] and 
                vol_spike[i] and 
                uptrend[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume in downtrend
            elif (close[i] < s1_aligned[i] and 
                  vol_spike[i] and 
                  downtrend[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R1 or trend changes
            if close[i] < r1_aligned[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1 or trend changes
            if close[i] > s1_aligned[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_Volume_TrendFilter"
timeframe = "1d"
leverage = 1.0