#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Signal_Trader
Hypothesis: Weekly pivot levels act as key institutional reference points.
Price tends to reverse from extreme levels (R4/S4) but shows continuation 
when breaking R3/S3 with volume confirmation. Uses tight entry conditions
(15-25 trades/year) to avoid fee drag and works in both bull/bear markets.
"""

name = "6h_Weekly_Pivot_Signal_Trader"
timeframe = "6h"
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
    
    # Get daily data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate weekly pivot points from daily data
    pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_pivot_points(
        daily_high, daily_low, daily_close
    )
    
    # Align weekly pivot levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_daily, r3)
    r4_6h = align_htf_to_ltf(prices, df_daily, r4)
    s3_6h = align_htf_to_ltf(prices, df_daily, s3)
    s4_6h = align_htf_to_ltf(prices, df_daily, s4)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # MEAN REVERSION LONG: Price at S4 with volume confirmation
            if close[i] <= s4_6h[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # MEAN REVERSION SHORT: Price at R4 with volume confirmation
            elif close[i] >= r4_6h[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            # BREAKOUT LONG: Price breaks above R3 with volume confirmation
            elif close[i] > r3_6h[i] and close[i-1] <= r3_6h[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # BREAKOUT SHORT: Price breaks below S3 with volume confirmation
            elif close[i] < s3_6h[i] and close[i-1] >= s3_6h[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R3 (take profit) or breaks below S3 (stop)
            if close[i] >= r3_6h[i] or close[i] < s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S3 (take profit) or breaks above R3 (stop)
            if close[i] <= s3_6h[i] or close[i] > r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals