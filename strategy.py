#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Squeeze_Breakout
Hypothesis: Breakouts from weekly pivot range (S3 to R3) with Bollinger Band squeeze and volume confirmation. Uses weekly pivot levels as structural support/resistance and Bollinger Band width to identify low-volatility compression before breakouts. Designed for 6h timeframe to capture multi-day moves with fewer trades, working in both bull and bear markets by requiring volatility contraction before directional moves.
"""

name = "6h_Weekly_Pivot_Squeeze_Breakout"
timeframe = "6h"
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
    
    # Get weekly data for pivot levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points and S3/R3 levels
    prev_weekly_high = df_weekly['high'].shift(1).values
    prev_weekly_low = df_weekly['low'].shift(1).values
    prev_weekly_close = df_weekly['close'].shift(1).values
    
    valid_idx = ~np.isnan(prev_weekly_high) & ~np.isnan(prev_weekly_low) & ~np.isnan(prev_weekly_close)
    pivot_point = np.full_like(prev_weekly_close, np.nan)
    weekly_r3 = np.full_like(prev_weekly_close, np.nan)
    weekly_s3 = np.full_like(prev_weekly_close, np.nan)
    
    # Standard pivot point calculation
    pivot_point[valid_idx] = (prev_weekly_high[valid_idx] + prev_weekly_low[valid_idx] + prev_weekly_close[valid_idx]) / 3.0
    weekly_range = prev_weekly_high[valid_idx] - prev_weekly_low[valid_idx]
    
    # S3 and R3 levels (more extreme levels)
    weekly_r3[valid_idx] = pivot_point[valid_idx] + weekly_range[valid_idx] * 1.1
    weekly_s3[valid_idx] = pivot_point[valid_idx] - weekly_range[valid_idx] * 1.1
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot_point)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    
    # Bollinger Band squeeze detection (20-period, 2 std dev)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized width
    
    # Squeeze condition: BB width below 20-period mean (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if weekly levels not available
        if np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with squeeze and volume confirmation
            if (high[i] > weekly_r3_aligned[i] and 
                squeeze_condition[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with squeeze and volume confirmation
            elif (low[i] < weekly_s3_aligned[i] and 
                  squeeze_condition[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point or squeeze breaks down
            if (low[i] <= pivot_aligned[i] or 
                not squeeze_condition[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point or squeeze breaks down
            if (high[i] >= pivot_aligned[i] or 
                not squeeze_condition[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals