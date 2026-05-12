#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_Filter
# Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with 1d EMA trend filter.
# Long when Bull Power > 0 and Bear Power < 0 (both bullish) + price > 1d EMA50.
# Short when Bull Power < 0 and Bear Power > 0 (both bearish) + price < 1d EMA50.
# Uses 6-period smoothing to reduce noise. Targets 15-30 trades/year to avoid fee drag.
# Works in bull via trend-following longs, in bear via trend-following shorts.

name = "6h_ElderRay_BullBearPower_1dTrend_Filter"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate EMA13 for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low

    # Smooth the power lines (6-period EMA) to reduce whipsaws
    bull_power_smooth = pd.Series(bull_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=6, adjust=False, min_periods=6).mean().values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):
        # Skip if any required value is NaN
        if (np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 (both bullish) + price > 1d EMA50
            if (bull_power_smooth[i] > 0 and 
                bear_power_smooth[i] < 0 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bull Power < 0 AND Bear Power > 0 (both bearish) + price < 1d EMA50
            elif (bull_power_smooth[i] < 0 and 
                  bear_power_smooth[i] > 0 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Either power signal turns bearish OR price crosses below 1d EMA50
            if (bull_power_smooth[i] <= 0 or bear_power_smooth[i] >= 0 or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Either power signal turns bullish OR price crosses above 1d EMA50
            if (bull_power_smooth[i] >= 0 or bear_power_smooth[i] <= 0 or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals