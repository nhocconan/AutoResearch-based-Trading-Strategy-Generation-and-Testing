#!/usr/bin/env python3
# 6h_SqueezeBreakout_WeeklyPivot
# Hypothesis: Combines Bollinger Band squeeze (low volatility) with weekly pivot breakout
# to capture explosive moves after consolidation. Uses weekly pivot levels as structural
# support/resistance and Bollinger Band width to identify low-volatility regimes.
# Works in both bull and bear markets: breakouts in either direction are traded.
# Bollinger squeeze identifies when volatility is compressed, increasing probability of
# a strong breakout. Weekly pivot provides meaningful levels that often act as
# breakout/breakdown zones. Volume confirmation filters out false breakouts.

name = "6h_SqueezeBreakout_WeeklyPivot"
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
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_6h = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_6h = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    r2_6h = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    s2_6h = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    r3_6h = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    s3_6h = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    
    # Bollinger Bands (20, 2) on 6h timeframe
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized width
    
    # Bollinger Squeeze: width below 20-period average width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume confirmation: volume above 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(squeeze[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price above R1 with squeeze and volume
            if close[i] > r1_6h[i] and squeeze[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price below S1 with squeeze and volume
            elif close[i] < s1_6h[i] and squeeze[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price below pivot or squeeze breaks (volatility expansion)
            if close[i] < pivot_6h[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price above pivot or squeeze breaks
            if close[i] > pivot_6h[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals