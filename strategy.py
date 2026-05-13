#!/usr/bin/env python3
# 1d_WeeklyPivot_R1_S1_Breakout_Trend_Filter_Volume
# Hypothesis: Weekly pivot resistance/support levels act as key institutional barriers.
# Breakouts above R1 or below S1 with volume confirmation and weekly trend filter capture
# institutional flow. Weekly trend (EMA50) ensures alignment with higher-timeframe momentum.
# Works in bull (buys R1 breakouts in uptrend) and bear (sells S1 breakdowns in downtrend).
# Target: 30-100 total trades over 4 years = 7-25/year.

name = "1d_WeeklyPivot_R1_S1_Breakout_Trend_Filter_Volume"
timeframe = "1d"
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

    # Get weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L
    # S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align weekly pivot levels to daily
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume filter: >1.8x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 + price above weekly EMA50 (bullish trend) + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + price below weekly EMA50 (bearish trend) + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 or price below weekly EMA50
            if (close[i] < s1_aligned[i] or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or price above weekly EMA50
            if (close[i] > r1_aligned[i] or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals