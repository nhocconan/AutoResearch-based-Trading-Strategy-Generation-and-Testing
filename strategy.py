#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_R3S3_Breakout_Trend_Volume
Hypothesis: Camarilla Pivot R3/S3 breakouts with 1-week trend filter and volume confirmation capture institutional breakouts in both bull and bear markets.
Breakouts above R3 + uptrend = long; breakdowns below S3 + downtrend = short.
Camarilla levels are derived from prior day's range and act as strong support/resistance.
Weekly trend filter avoids counter-trend trades. Volume confirmation ensures institutional participation.
Target: 15-25 trades/year per symbol with disciplined risk management.
"""

name = "1d_Camarilla_Pivot_R3S3_Breakout_Trend_Volume"
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

    # Get 1-week data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # 1-week EMA20 for trend
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Calculate Camarilla levels from previous day's range
    # R3 = close + 1.1 * (high - low) * 1.1
    # S3 = close - 1.1 * (high - low) * 1.1
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1

    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        if np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above Camarilla R3 + weekly uptrend + volume spike
            if close[i] > camarilla_r3[i] and close[i] > ema20_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Camarilla S3 + weekly downtrend + volume spike
            elif close[i] < camarilla_s3[i] and close[i] < ema20_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below Camarilla Pivot Point (CP) or weekly trend turns down
            cp = (prev_high[i] + prev_low[i] + prev_close[i]) / 3
            if close[i] < cp or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above Camarilla Pivot Point (CP) or weekly trend turns up
            cp = (prev_high[i] + prev_low[i] + prev_close[i]) / 3
            if close[i] > cp or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals