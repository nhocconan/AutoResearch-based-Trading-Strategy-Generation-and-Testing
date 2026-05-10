#!/usr/bin/env python3
# 1d_Weekly_Pivot_Breakout_Momentum
# Hypothesis: Use weekly pivot points on 1d timeframe with momentum confirmation and volume filter.
# Weekly pivots provide strong support/resistance levels. Price breaking above/below these levels
# with momentum (price > 20-period EMA) and volume confirmation indicates institutional interest.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at pivots).
# Targets 8-20 trades/year to minimize fee drag.

name = "1d_Weekly_Pivot_Breakout_Momentum"
timeframe = "1d"
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
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift to get previous week's values (avoid look-ahead)
    prev_high = np.concatenate([[high_1w[0]], high_1w[:-1]])
    prev_low = np.concatenate([[low_1w[0]], low_1w[:-1]])
    prev_close = np.concatenate([[close_1w[0]], close_1w[:-1]])
    
    # Weekly pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Align weekly levels to daily
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Daily EMA20 for momentum filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5 * 20-day average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema20[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above weekly R1 with upward momentum and volume
            if (close[i] > r1_aligned[i] and
                close[i] > ema20[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with downward momentum and volume
            elif (close[i] < s1_aligned[i] and
                  close[i] < ema20[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to weekly pivot or momentum fails
            if (close[i] < pivot_aligned[i] or
                close[i] < ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to weekly pivot or momentum fails
            if (close[i] > pivot_aligned[i] or
                close[i] > ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals