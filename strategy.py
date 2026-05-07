#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: 12h chart strategy using weekly pivot-based R1/S1 breakouts filtered by 1w EMA50 trend and volume confirmation (1.5x average volume).
# Weekly R1/S1 act as strong support/resistance with high probability of reversal or breakout.
# 1w EMA50 provides trend filter to avoid counter-trend trades. Volume confirms breakout validity.
# Designed to work in both bull and bear markets by filtering with trend and requiring volume confirmation.
# Target: 15-30 trades/year per symbol to minimize fee drag while maintaining edge.

timeframe = "12h"
name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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
    
    # Get weekly data for pivot points (R1, S1) and trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly pivot points: R1, S1
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_point = (high_1w + low_1w + close_1w) / 3
    pivot_r1 = 2 * pivot_point - low_1w
    pivot_s1 = 2 * pivot_point - high_1w
    
    pivot_r1_aligned = align_htf_to_ltf(prices, df_1w, pivot_r1)
    pivot_s1_aligned = align_htf_to_ltf(prices, df_1w, pivot_s1)
    
    # Volume spike detection: 1.5x average volume (2-period = 1 day on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2)  # Ensure we have EMA50 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(pivot_r1_aligned[i]) or 
            np.isnan(pivot_s1_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume, and 1w trend is bullish (price > EMA50)
            if (high[i] > pivot_r1_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume, and 1w trend is bearish (price < EMA50)
            elif (low[i] < pivot_s1_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below weekly S1 (reversal signal)
            if low[i] < pivot_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above weekly R1 (reversal signal)
            if high[i] > pivot_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals