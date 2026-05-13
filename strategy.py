#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Volume_Squeeze
Hypothesis: Weekly pivot levels (S4/R4) act as extreme support/resistance where price mean-reverts during low volatility (squeeze), while breakouts above R3/below S3 during high volatility (expansion) continue the trend. Volume and Bollinger Band width filter regimes. Designed for 15-25 trades/year to work in bull/bear markets by capturing reversals at extremes and breakouts in strong moves.
"""

name = "6h_Weekly_Pivot_Volume_Squeeze"
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
    
    # Get weekly data (using daily as proxy for weekly calculation)
    df_weekly = get_htf_data(prices, '1d')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points
    pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_pivot_points(
        weekly_high, weekly_low, weekly_close
    )
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_weekly, pivot)
    r3_6h = align_htf_to_ltf(prices, df_weekly, r3)
    r4_6h = align_htf_to_ltf(prices, df_weekly, r4)
    s3_6h = align_htf_to_ltf(prices, df_weekly, s3)
    s4_6h = align_htf_to_ltf(prices, df_weekly, s4)
    
    # Bollinger Band width for volatility regime (20-period, 2 std)
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_width = (4 * bb_std) / bb_mid  # (upper - lower) / middle
    bb_width = bb_width.fillna(0).values
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # MEAN REVERSION LONG: Price at S4 during low volatility (squeeze)
            if close[i] <= s4_6h[i] and bb_width[i] < np.percentile(bb_width[:i+1], 30) and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # MEAN REVERSION SHORT: Price at R4 during low volatility (squeeze)
            elif close[i] >= r4_6h[i] and bb_width[i] < np.percentile(bb_width[:i+1], 30) and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            # BREAKOUT LONG: Price breaks above R3 during high volatility (expansion)
            elif close[i] > r3_6h[i] and close[i-1] <= r3_6h[i-1] and bb_width[i] > np.percentile(bb_width[:i+1], 70) and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # BREAKOUT SHORT: Price breaks below S3 during high volatility (expansion)
            elif close[i] < s3_6h[i] and close[i-1] >= s3_6h[i-1] and bb_width[i] > np.percentile(bb_width[:i+1], 70) and volume_confirm[i]:
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