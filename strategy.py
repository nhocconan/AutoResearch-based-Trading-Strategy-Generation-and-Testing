#!/usr/bin/env python3
# 1d_WeeklyTrend_DailyBreakout_Volume
# Hypothesis: Daily breakout from weekly high/low with weekly trend filter and volume confirmation.
# Uses weekly high/low for breakout levels and weekly EMA50 for trend direction.
# Volume confirmation: daily volume > 2.0 x 20-day average.
# Designed to capture strong breakouts in the direction of weekly trend.
# Target: 7-25 trades/year per symbol to minimize fee decay while maintaining edge.

name = "1d_WeeklyTrend_DailyBreakout_Volume"
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

    # Get weekly data for trend filter and breakout levels
    df_weekly = get_htf_data(prices, '1w')

    # Weekly high/low for breakout levels (using previous week)
    weekly_high = df_weekly['high'].shift(1).values
    weekly_low = df_weekly['low'].shift(1).values

    # Weekly EMA50 for trend filter
    weekly_close = df_weekly['close'].values
    ema50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Align weekly indicators to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)

    # Volume confirmation: daily volume > 2.0 x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema50_weekly_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above weekly high in uptrend with volume spike
            if (close[i] > weekly_high_aligned[i] and 
                close[i] > ema50_weekly_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly low in downtrend with volume spike
            elif (close[i] < weekly_low_aligned[i] and 
                  close[i] < ema50_weekly_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly low or trend turns down
            if close[i] < weekly_low_aligned[i] or close[i] < ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly high or trend turns up
            if close[i] > weekly_high_aligned[i] or close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals