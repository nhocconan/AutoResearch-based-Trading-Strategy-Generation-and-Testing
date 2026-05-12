#!/usr/bin/env python3
# 6h_PivotZone_MeanReversion
# Hypothesis: Fade price from daily pivot support/resistance zones (S1/S2, R1/R2) with
# 1d trend filter and volume confirmation. In bull markets, buy dips to S1/S2 in uptrend;
# in bear markets, sell rallies to R1/R2 in downtrend. Uses mean reversion within
# the dominant trend to avoid counter-trend whipsaws. Low frequency via zone-based
# entries (not single-level breaks) and trend alignment.

name = "6h_PivotZone_MeanReversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """
    Calculate classic pivot points and support/resistance levels.
    Pivot = (H + L + C) / 3
    R1 = 2*Pivot - L
    S1 = 2*Pivot - H
    R2 = Pivot + (H - L)
    S2 = Pivot - (H - L)
    Returns arrays for Pivot, R1, S1, R2, S2.
    """
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points
    pivot_1d, r1_1d, s1_1d, r2_1d, s2_1d = calculate_pivot_points(high_1d, low_1d, close_1d)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align daily data to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 is stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price relative to daily EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_24[i]
        
        # Proximity to pivot zones (within 0.5% of level)
        def near_level(price, level, threshold=0.005):
            return abs(price - level) / level < threshold
        
        near_s1 = near_level(close[i], s1_1d_aligned[i])
        near_s2 = near_level(close[i], s2_1d_aligned[i])
        near_r1 = near_level(close[i], r1_1d_aligned[i])
        near_r2 = near_level(close[i], r2_1d_aligned[i])
        
        if position == 0:
            # LONG: near S1/S2 in uptrend with volume
            if (near_s1 or near_s2) and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: near R1/R2 in downtrend with volume
            elif (near_r1 or near_r2) and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price reaches pivot or trend fails
            if close[i] >= pivot_1d_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches pivot or trend fails
            if close[i] <= pivot_1d_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals