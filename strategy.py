#!/usr/bin/env python3
# 6h_WeeklyPivot_RangeBreakout_TrendFilter
# Hypothesis: Price breaks above/below weekly pivot levels with trend confirmation from 1d EMA34 and volume >1.5x 20-bar average.
# Weekly pivot levels provide strong support/resistance that work in both bull and bear markets.
# Trend filter ensures we only take breaks in the direction of higher timeframe trend.
# Volume confirmation ensures only high-conviction breakouts trigger entries.
# Designed for 15-30 trades/year on 6h timeframe to minimize fee drag.

name = "6h_WeeklyPivot_RangeBreakout_TrendFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = np.full_like(close_1w, np.nan)
    valid_1w = ~np.isnan(high_1w) & ~np.isnan(low_1w) & ~np.isnan(close_1w)
    pivot_1w[valid_1w] = (high_1w[valid_1w] + low_1w[valid_1w] + close_1w[valid_1w]) / 3.0
    
    # Calculate support and resistance levels
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1w = np.full_like(close_1w, np.nan)
    s1_1w = np.full_like(close_1w, np.nan)
    r1_1w[valid_1w] = 2 * pivot_1w[valid_1w] - low_1w[valid_1w]
    s1_1w[valid_1w] = 2 * pivot_1w[valid_1w] - high_1w[valid_1w]
    
    # R2 = P + (H - L), S2 = P - (H - L)
    r2_1w = np.full_like(close_1w, np.nan)
    s2_1w = np.full_like(close_1w, np.nan)
    r2_1w[valid_1w] = pivot_1w[valid_1w] + (high_1w[valid_1w] - low_1w[valid_1w])
    s2_1w[valid_1w] = pivot_1w[valid_1w] - (high_1w[valid_1w] - low_1w[valid_1w])
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 32) / 34
    
    # Align 1d EMA to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: 6h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or \
           np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or \
           np.isnan(s2_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or \
           np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above R1 with volume confirmation AND bullish trend
            if close[i] > r1_1w_aligned[i] and volume_ratio[i] > 1.5 and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S1 with volume confirmation AND bearish trend
            elif close[i] < s1_1w_aligned[i] and volume_ratio[i] > 1.5 and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below S1 (reversal) or trend turns bearish
            if close[i] < s1_1w_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above R1 (reversal) or trend turns bullish
            if close[i] > r1_1w_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals