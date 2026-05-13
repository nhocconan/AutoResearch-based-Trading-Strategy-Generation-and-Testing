#!/usr/bin/env python3
# 6h_WeeklyPivot_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use weekly pivot R1/S1 levels for breakout entries on 6h timeframe, filtered by 1-day EMA trend and volume confirmation.
# Weekly pivots provide strong institutional support/resistance levels. Breakouts from these levels often carry momentum.
# The 1d EMA filter ensures trades align with the daily trend, reducing false breakouts in choppy markets.
# Volume confirmation adds conviction to breakout moves.
# Works in bull markets (follows upward breaks with bullish daily trend) and bear markets (avoids upward breaks in bearish daily trend, takes downward breaks).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_WeeklyPivot_R1_S1_Breakout_1dTrend_Volume"
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

    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (R1, S1)
    # Standard pivot point: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    pivot_point = (high_weekly + low_weekly + close_weekly) / 3
    weekly_r1 = 2 * pivot_point - low_weekly
    weekly_s1 = 2 * pivot_point - high_weekly
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend filter
    ema_50_daily = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly and daily indicators to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)

    # Volume filter: >1.5x 20-period average on 6h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema_50_daily_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above weekly R1 + price above daily EMA50 (bullish trend) + volume spike
            if (close[i] > weekly_r1_aligned[i] and 
                close[i] > ema_50_daily_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below weekly S1 + price below daily EMA50 (bearish trend) + volume spike
            elif (close[i] < weekly_s1_aligned[i] and 
                  close[i] < ema_50_daily_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below weekly S1 or price below daily EMA50
            if (close[i] < weekly_s1_aligned[i] or close[i] < ema_50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above weekly R1 or price above daily EMA50
            if (close[i] > weekly_r1_aligned[i] or close[i] > ema_50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals