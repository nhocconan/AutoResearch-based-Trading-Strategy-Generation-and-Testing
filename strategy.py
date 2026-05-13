#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_Trend_Filter
Hypothesis: Combines weekly pivot points (from 1w data) as key support/resistance levels with trend filtering from 1d EMA to capture breakouts in both bull and bear markets. Weekly pivots provide institutional-grade levels that hold across market regimes, while EMA filter ensures trades align with higher timeframe momentum. Designed for low trade frequency (15-30/year) with clear breakout logic.
"""

name = "6h_Weekly_Pivot_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day EMA for trend filter (50 period)
    ema_50_1d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: (H + L + C) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pivot_points = typical_price.values
    
    # Calculate support and resistance levels
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    r1 = 2 * pivot_points - low_1w
    s1 = 2 * pivot_points - high_1w
    r2 = pivot_points + (high_1w - low_1w)
    s2 = pivot_points - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot_points - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot_points)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_points)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price breaks above R3 with volume confirmation and uptrend filter
            if close[i] > r3_aligned[i] and volume_confirm[i] and close[i] > ema_50_1d[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume confirmation and downtrend filter
            elif close[i] < s3_aligned[i] and volume_confirm[i] and close[i] < ema_50_1d[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below pivot point or S1 (strong support break)
            if close[i] < pivot_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above pivot point or R1 (strong resistance break)
            if close[i] > pivot_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals