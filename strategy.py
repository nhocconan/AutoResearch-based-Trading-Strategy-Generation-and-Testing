#!/usr/bin/env python3
# 1d_Pivot_Bounce_With_WeeklyTrend
# Hypothesis: Use weekly trend filter + daily Camarilla pivot levels for mean reversion.
# In uptrend (price > weekly EMA50), buy at S3/S4 support with volume confirmation.
# In downtrend (price < weekly EMA50), sell at R3/R4 resistance with volume confirmation.
# Exit at opposite pivot level or when trend reversals. Designed for low trade frequency
# (<20/year) to minimize fee drag, works in both bull and bear via trend-following mean reversion.

name = "1d_Pivot_Bounce_With_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_levels(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Calculate daily Camarilla pivot levels
    r3, r4, s3, s4 = calculate_pivot_levels(high, low, close)

    # Volume confirmation: current volume > 1.3x average of last 10 days
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(r4[i]) or np.isnan(s3[i]) or np.isnan(s4[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from weekly EMA50
        price_above_weekly_ema = close[i] > ema_50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_50_1w_aligned[i]

        if position == 0:
            # LONG: In uptrend, price touches S3/S4 support with volume
            if price_above_weekly_ema and (close[i] <= s3[i] or close[i] <= s4[i]) and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: In downtrend, price touches R3/R4 resistance with volume
            elif price_below_weekly_ema and (close[i] >= r3[i] or close[i] >= r4[i]) and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R3/R4 resistance OR trend turns down
            if (close[i] >= r3[i] or close[i] >= r4[i]) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S3/S4 support OR trend turns up
            if (close[i] <= s3[i] or close[i] <= s4[i]) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals