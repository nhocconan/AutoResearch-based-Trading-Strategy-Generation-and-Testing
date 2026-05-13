#!/usr/bin/env python3
# 6h_WeeklyPivot_MonthlyTrend_Breakout
# Hypothesis: Weekly pivot levels (R1/S1) from Monday's open act as key support/resistance.
# Breakouts above R1 with monthly (1M) EMA50 uptrend and volume confirmation capture momentum in bull markets.
# Breakdowns below S1 with monthly EMA50 downtrend capture bearish moves.
# Monthly trend filter reduces whipsaws in sideways markets, while weekly pivots adapt to changing volatility.
# Target: 15-35 trades per year per symbol to minimize fee drag and ensure robustness.

name = "6h_WeeklyPivot_MonthlyTrend_Breakout"
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

    # Get weekly data for pivot calculation (using Monday's open as week start)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot from previous week's OHLC
    # Using Friday's close as weekly close approximation
    prev_weekly_close = df_weekly['close'].shift(1).values
    prev_weekly_high = df_weekly['high'].shift(1).values
    prev_weekly_low = df_weekly['low'].shift(1).values
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    weekly_r1 = weekly_pivot + (prev_weekly_high - prev_weekly_low) * 1.1 / 12
    weekly_s1 = weekly_pivot - (prev_weekly_high - prev_weekly_low) * 1.1 / 12
    
    # Align weekly pivot to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)

    # Get monthly data for trend filter
    df_monthly = get_htf_data(prices, '1M')
    
    # Monthly EMA50 for trend filter
    ema50_monthly = pd.Series(df_monthly['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_monthly_aligned = align_htf_to_ltf(prices, df_monthly, ema50_monthly)

    # Volume filter: >1.5x 24-period average (4 days of 6h bars)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema50_monthly_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above weekly R1 + monthly EMA50 uptrend + volume spike
            if (close[i] > weekly_r1_aligned[i] and 
                close[i] > ema50_monthly_aligned[i] and
                volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below weekly S1 + monthly EMA50 downtrend + volume spike
            elif (close[i] < weekly_s1_aligned[i] and 
                  close[i] < ema50_monthly_aligned[i] and
                  volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly S1 or trend reversal
            if close[i] < weekly_s1_aligned[i] or close[i] < ema50_monthly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly R1 or trend reversal
            if close[i] > weekly_r1_aligned[i] or close[i] > ema50_monthly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals