#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_WeeklyTrend
# Hypothesis: Fade at Camarilla R3/S3 levels when aligned with weekly trend, with volume confirmation.
# In bull markets, weekly trend favors long at S3; in bear markets, weekly trend favors short at R3.
# Uses weekly EMA50 for trend direction and volume spike for confirmation.
# Target: 15-25 trades/year on 6h to stay within optimal range while capturing reversal opportunities.

name = "6h_Camarilla_R3_S3_Breakout_WeeklyTrend"
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

    # Calculate weekly trend using EMA50 on weekly data
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_trend = align_htf_to_ltf(prices, df_weekly, weekly_ema50)

    # Calculate daily Camarilla pivot levels
    df_daily = get_htf_data(prices, '1d')
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Camarilla pivot calculation
    camarilla_pivot = (daily_high + daily_low + daily_close) / 3
    daily_range = daily_high - daily_low
    r3 = camarilla_pivot + daily_range * 1.1 / 2
    s3 = camarilla_pivot - daily_range * 1.1 / 2
    
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(weekly_trend[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at S3 support with weekly uptrend + volume spike
            if (close[i] <= s3_aligned[i] and 
                close[i] > weekly_trend[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R3 resistance with weekly downtrend + volume spike
            elif (close[i] >= r3_aligned[i] and 
                  close[i] < weekly_trend[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above weekly EMA50 (trend change)
            if close[i] >= weekly_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below weekly EMA50 (trend change)
            if close[i] <= weekly_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals