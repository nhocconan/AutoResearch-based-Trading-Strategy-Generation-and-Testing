#!/usr/bin/env python3
"""
6h_WeeklyPivot_Confluence_1dTrend_Volume
Hypothesis: On 6h timeframe, use weekly pivot points (S1/S2/S3/R1/R2/R3) for mean reversion entries when price reaches support/resistance, filtered by 1d trend (price above/below EMA50) and volume confirmation. This captures reversals at institutional levels while avoiding counter-trend trades. Weekly pivots provide structure, 1d trend ensures we trade with momentum, and volume filters out weak moves. Designed for 15-35 trades/year to minimize fee drag in both bull and bear markets.
"""

name = "6h_WeeklyPivot_Confluence_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L, etc.
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    s1 = 2 * pivot - weekly_high
    s2 = pivot - (weekly_high - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    r1 = 2 * pivot - weekly_low
    r2 = pivot + (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 1d data for volume filter
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 6h data for price
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivots and 1d indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1d EMA50
        uptrend_1d = close[i] > ema50_1d_aligned[i]
        downtrend_1d = close[i] < ema50_1d_aligned[i]
        
        # Volume filter: current 6h volume > 1.3x 1d 20-period MA
        volume_filter = volume[i] > vol_ma20_1d_aligned[i] * 1.3
        
        # Proximity to pivot levels (within 0.3% for entry)
        proximity_threshold = 0.003
        near_s1 = abs(close[i] - s1_aligned[i]) / close[i] <= proximity_threshold
        near_s2 = abs(close[i] - s2_aligned[i]) / close[i] <= proximity_threshold
        near_s3 = abs(close[i] - s3_aligned[i]) / close[i] <= proximity_threshold
        near_r1 = abs(close[i] - r1_aligned[i]) / close[i] <= proximity_threshold
        near_r2 = abs(close[i] - r2_aligned[i]) / close[i] <= proximity_threshold
        near_r3 = abs(close[i] - r3_aligned[i]) / close[i] <= proximity_threshold
        
        if position == 0:
            # Long: price near support in uptrend with volume
            if (near_s1 or near_s2 or near_s3) and uptrend_1d and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price near resistance in downtrend with volume
            elif (near_r1 or near_r2 or near_r3) and downtrend_1d and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches resistance or trend fails
            if (near_r1 or near_r2 or near_r3) or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches support or trend fails
            if (near_s1 or near_s2 or near_s3) or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals