#!/usr/bin/env python3
"""
6h_WeeklyPivot_1dTrend_WithVolume
Hypothesis: Weekly pivot points provide strong support/resistance that hold across both bull and bear markets.
In bull markets: long when price breaks above weekly R1 with 1d uptrend and volume confirmation.
In bear markets: short when price breaks below weekly S1 with 1d downtrend and volume confirmation.
Uses weekly pivots (H/L/C from prior week) and 1d EMA50 trend filter.
Target: 50-150 total trades over 4 years (~12-37/year) to balance opportunity and cost.
"""

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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    pivot = (high_w + low_w + close_w) / 3
    weekly_r1 = 2 * pivot - low_w
    weekly_s1 = 2 * pivot - high_w
    
    # Align weekly pivots to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA to 6h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # ATR for volatility measurement
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(atr_period, vol_ma_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above weekly R1 with uptrend and volume
            if uptrend and volume_confirmation and price > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with downtrend and volume
            elif downtrend and volume_confirmation and price < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below weekly pivot or trend reverses
            if price < pivot[min(i // len(df_weekly), len(pivot)-1)] if len(pivot) > 0 else r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above weekly pivot or trend reverses
            if price > pivot[min(i // len(df_weekly), len(pivot)-1)] if len(pivot) > 0 else s1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_1dTrend_WithVolume"
timeframe = "6h"
leverage = 1.0