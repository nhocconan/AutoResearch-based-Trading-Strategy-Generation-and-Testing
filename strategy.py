#!/usr/bin/env python3
name = "6h_WeeklyPivotBias_VolumeSpike_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Weekly pivot levels (resistance/support)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    # Use previous week's close for pivot calculation (avoid look-ahead)
    prev_week_close = df_1w['close'].values
    # Calculate pivot points using previous week's OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Resistance 1 = (2 * P) - L
    r1 = (2 * pivot) - weekly_low
    # Support 1 = (2 * P) - H
    s1 = (2 * pivot) - weekly_high
    # Resistance 2 = P + (H - L)
    r2 = pivot + (weekly_high - weekly_low)
    # Support 2 = P - (H - L)
    s2 = pivot - (weekly_high - weekly_low)
    # Resistance 3 = H + 2*(P - L)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    # Support 3 = L - 2*(H - P)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and 1d uptrend
            if close[i] > r1_aligned[i] and volume_filter[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and 1d downtrend
            elif close[i] < s1_aligned[i] and volume_filter[i] and not trend_up[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 or 1d trend turns down
            if close[i] < s1_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R1 or 1d trend turns up
            if close[i] > r1_aligned[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals