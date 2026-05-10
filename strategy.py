#!/usr/bin/env python3
"""
4h_VolumeBreakout_VolatilityFilter
Hypothesis: Strong volume expansion (>2x 1d volume) combined with volatility contraction (ATR ratio < 0.6) precedes breakouts. 
Enter long when price breaks above 4h high of last 20 bars, short when breaks below low, only in direction of 1d EMA50 trend.
Exit when price crosses 10-bar EMA or volatility expands (ATR ratio > 1.2). 
Designed for 4-6 trades/year per side, ~16-24 total over 4 years to minimize fee drag.
Works in bull/bear by trend filter and volatility regime detection.
"""

name = "4h_VolumeBreakout_VolatilityFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # 1d ATR for volatility regime (14-period)
    atr14_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        tr1 = high_1d[1:] - low_1d[:-1]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum.reduce([tr1, tr2, tr3])
        tr = np.concatenate([[np.nan], tr])  # align length
        atr14_1d[13] = np.nanmean(tr[1:15])  # first valid at index 13
        for i in range(14, len(tr)):
            atr14_1d[i] = (atr14_1d[i-1] * 13 + tr[i]) / 14
    
    # 4h ATR for current volatility (14-period)
    tr1_4h = high[1:] - low[:-1]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.maximum.reduce([tr1_4h, tr2_4h, tr3_4h])
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr14_4h = np.full(n, np.nan)
    if n >= 14:
        atr14_4h[13] = np.nanmean(tr_4h[1:15])
        for i in range(14, n):
            atr14_4h[i] = (atr14_4h[i-1] * 13 + tr_4h[i]) / 14
    
    # 4h highest high/lowest low of last 20 bars (Donchian-like)
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # Align 1d indicators to 4h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA50 and 20-bar lookback
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or np.isnan(volume_1d_aligned[i]) or \
           np.isnan(atr14_4h[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume expansion: current 4h volume > 2x average 1d volume (scaled to 4h)
        vol_1d_scaled = volume_1d_aligned[i] / 6.0  # 6x 4h bars in 1d
        volume_expansion = volume[i] > 2.0 * vol_1d_scaled
        
        # Volatility contraction: current 4h ATR < 0.6x 1d ATR (low volatility regime)
        vol_contraction = atr14_4h[i] < 0.6 * atr14_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > highest_20[i]
        breakout_down = close[i] < lowest_20[i]
        
        # Trend filter
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: volatility contraction + volume expansion + upward breakout + uptrend
            if vol_contraction and volume_expansion and breakout_up and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: volatility contraction + volume expansion + downward breakout + downtrend
            elif vol_contraction and volume_expansion and breakout_down and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend fails OR volatility expands (>1.2x) OR price crosses 10-bar EMA
            ema10 = np.mean(close[i-9:i+1]) if i >= 9 else close[i]  # simple 10-bar average
            vol_expansion = atr14_4h[i] > 1.2 * atr14_1d_aligned[i]
            if not is_uptrend or vol_expansion or close[i] < ema10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend fails OR volatility expands OR price crosses 10-bar EMA
            ema10 = np.mean(close[i-9:i+1]) if i >= 9 else close[i]
            vol_expansion = atr14_4h[i] > 1.2 * atr14_1d_aligned[i]
            if not is_downtrend or vol_expansion or close[i] > ema10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals