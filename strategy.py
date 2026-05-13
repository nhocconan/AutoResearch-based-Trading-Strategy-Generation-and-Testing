#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use 4h Camarilla R1/S1 breakout with 1d EMA trend filter and volume confirmation.
# Camarilla levels provide intraday support/resistance; breakout with trend filter captures momentum.
# Works in bull (follows breakouts in uptrend) and bear (avoids counter-trend breakouts).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Get 4h data for Camarilla calculation (using typical price)
    df_4h = get_htf_data(prices, '4h')
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3.0
    # Camarilla R1 and S1 from previous day's typical price
    # Using 1d data to get previous day's high, low, close
    prev_day_high = df_4h['high'].rolling(window=6, min_periods=6).max().values  # 6*4h = 24h
    prev_day_low = df_4h['low'].rolling(window=6, min_periods=6).min().values
    prev_day_close = df_4h['close'].rolling(window=6, min_periods=6).mean().values
    # Camarilla formulas: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = prev_day_close + 1.1 * (prev_day_high - prev_day_low) / 12
    camarilla_s1 = prev_day_close - 1.1 * (prev_day_high - prev_day_low) / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 + price above 1d EMA (uptrend) + volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + price below 1d EMA (downtrend) + volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 or price below 1d EMA
            if (close[i] < camarilla_s1_aligned[i] or close[i] < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or price above 1d EMA
            if (close[i] > camarilla_r1_aligned[i] or close[i] > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals