#!/usr/bin/env python3
"""
12h_Weekly_Pivot_R3S3_Breakout_With_Volume
Hypothesis: Weekly pivot levels (R3/S3) act as strong support/resistance on 12h timeframe.
Price breaking above R3 with volume confirmation indicates bullish momentum; breaking below S3 indicates bearish momentum.
Designed for low trade frequency (12-30/year) to work in both bull and bear markets by capturing breakouts from key weekly levels.
"""

name = "12h_Weekly_Pivot_R3S3_Breakout_With_Volume"
timeframe = "12h"
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
    
    # Align weekly pivot levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_weekly, r3)
    s3_12h = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # BREAKOUT LONG: Price breaks above R3 with volume confirmation
            if close[i] > r3_12h[i] and close[i-1] <= r3_12h[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # BREAKOUT SHORT: Price breaks below S3 with volume confirmation
            elif close[i] < s3_12h[i] and close[i-1] >= s3_12h[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks back below R3 (failed breakout) or reaches R4 (take profit)
            if close[i] < r3_12h[i] or close[i] >= r4_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks back above S3 (failed breakdown) or reaches S4 (take profit)
            if close[i] > s3_12h[i] or close[i] <= s4_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Align R4 and S4 for exit conditions
    pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_pivot_points(
        weekly_high, weekly_low, weekly_close
    )
    
    # Align weekly pivot levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_weekly, r3)
    r4_12h = align_htf_to_ltf(prices, df_weekly, r4)
    s3_12h = align_htf_to_ltf(prices, df_weekly, s3)
    s4_12h = align_htf_to_ltf(prices, df_weekly, s4)