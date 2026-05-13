#!/usr/bin/env python3
"""
1d_MeanReversion_Extreme_Pivot_With_Volume
Hypothesis: Daily price extremes at weekly pivot R4/S4 levels with volume confirmation 
provide high-probability mean-reversion opportunities. Price tends to revert from 
these extreme levels in both bull and bear markets. The strategy uses weekly pivots 
as structural support/resistance and requires volume confirmation to filter false 
signals. Target: 10-25 trades/year with low turnover to minimize fee drag.
"""

name = "1d_MeanReversion_Extreme_Pivot_With_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points: P, R1-R4, S1-S4"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points
    pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_pivot_points(
        weekly_high, weekly_low, weekly_close
    )
    
    # Align weekly pivot levels to daily timeframe
    pivot_d = align_htf_to_ltf(prices, df_weekly, pivot)
    r4_d = align_htf_to_ltf(prices, df_weekly, r4)
    s4_d = align_htf_to_ltf(prices, df_weekly, s4)
    
    # Volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # MEAN REVERSION LONG: Price at S4 with volume confirmation
            if close[i] <= s4_d[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # MEAN REVERSION SHORT: Price at R4 with volume confirmation
            elif close[i] >= r4_d[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot (mean reversion complete)
            if close[i] >= pivot_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot (mean reversion complete)
            if close[i] <= pivot_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals