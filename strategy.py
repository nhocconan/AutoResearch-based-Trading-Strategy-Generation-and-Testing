#!/usr/bin/env python3

# 6H_1W_1D_Camarilla_R3S3_Breakout_Trend_Volume
# Hypothesis: Use weekly trend filter and daily Camarilla levels to identify breakouts on 6-hour timeframe.
# Enter long when price breaks above daily R3 with bullish weekly trend and volume spike.
# Enter short when price breaks below daily S3 with bearish weekly trend and volume spike.
# Exit when price crosses opposite Camarilla level (S3 for long, R3 for short) or trend reversals.
# Weekly trend ensures alignment with higher timeframe momentum, reducing false breakouts.
# Volume confirmation filters low-quality breakouts. Designed for low frequency (12-37 trades/year) to minimize fee drag.

name = "6H_1W_1D_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily Camarilla levels (based on previous day)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    # Handle first value
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    rang = prev_high - prev_low
    R3 = prev_close + rang * 1.1 / 4
    S3 = prev_close - rang * 1.1 / 4

    # Align daily Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)

    # Volume confirmation: current volume > 1.5x average of last 24 periods (4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Weekly trend filter
        bullish_trend = close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i]

        if position == 0:
            # LONG: Price crosses above R3 with bullish weekly trend and volume confirmation
            if close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below S3 with bearish weekly trend and volume spike
            elif close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3 or weekly trend turns bearish
            if close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 or weekly trend turns bullish
            if close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals