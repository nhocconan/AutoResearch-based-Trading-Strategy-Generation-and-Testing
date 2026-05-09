#!/usr/bin/env python3
# 6H_1W_1D_PivotBreakout_TrendFilter
# Hypothesis: On 6h timeframe, enter long when price breaks above weekly pivot R1 with 1d uptrend and volume confirmation.
# Short when price breaks below weekly pivot S1 with 1d downtrend and volume confirmation.
# Uses weekly pivot levels for structure and 1d trend to avoid counter-trend trades.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years).

name = "6H_1W_1D_PivotBreakout_TrendFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    pivot = (high_1w + low_1w + close_1w) / 3
    weekly_r1 = 2 * pivot - low_1w
    weekly_s1 = 2 * pivot - high_1w
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d trend: EMA(34) on close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1d > ema_34
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align weekly and daily indicators to 6h
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly R1 + 1d uptrend + volume confirmation
            if close[i] > weekly_r1_aligned[i] and trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S1 + 1d downtrend + volume confirmation
            elif close[i] < weekly_s1_aligned[i] and not trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly S1 (reversal) or trend changes
            if close[i] < weekly_s1_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly R1 (reversal) or trend changes
            if close[i] > weekly_r1_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals