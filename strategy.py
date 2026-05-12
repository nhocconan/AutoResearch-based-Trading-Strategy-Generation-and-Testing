#!/usr/bin/env python3
# 6h_1D_1W_Camarilla_R3S3_Breakout_Trend
# Hypothesis: Breakout above Camarilla R3 or below S3 with 1d trend filter (EMA34) and volume confirmation.
# Uses weekly pivot to filter direction: only take long if weekly pivot shows uptrend, short if downtrend.
# Targets 15-30 trades/year on 6f timeframe with strict entry conditions to avoid overtrading.
# Works in both bull and bear markets by requiring trend alignment and volume confirmation.

name = "6h_1D_1W_Camarilla_R3S3_Breakout_Trend"
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

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate Camarilla levels from previous 1d candle
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We only need R3 and S3 for breakout
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d

    # Calculate R3 and S3 for each 1d bar
    r3 = close_1d + 1.1 * range_1d
    s3 = close_1d - 1.1 * range_1d

    # Align Camarilla levels to 6h timeframe (using previous day's values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup for volume MA
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filters: price above/below EMA on 1d and 1w
        bullish_trend = close[i] > ema_1d_aligned[i] and close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i] and close[i] < ema_1w_aligned[i]

        if position == 0:
            # LONG: Price breaks above R3 with bullish trend and volume confirmation
            if close[i] > r3_aligned[i] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with bearish trend and volume confirmation
            elif close[i] < s3_aligned[i] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S3 or trend turns bearish
            if close[i] < s3_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R3 or trend turns bullish
            if close[i] > r3_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals