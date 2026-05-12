#!/usr/bin/env python3
# 1H_4H_1D_Camarilla_R1S1_Breakout_Trend_Filter
# Hypothesis: 1H strategy using 4H trend direction (EMA50) and 1D Camarilla R1/S1 breakouts with volume confirmation.
# Uses 4H EMA50 for trend direction, 1D Camarilla levels for breakout signals, and 1H for entry timing.
# Designed to work in bull markets (buy R1 breakout in uptrend) and bear markets (sell S1 breakdown in downtrend).
# Includes session filter (08-20 UTC) and volume confirmation to reduce false signals.
# Target: 15-37 trades/year on 1H timeframe to avoid fee drag.

name = "1H_4H_1D_Camarilla_R1S1_Breakout_Trend_Filter"
timeframe = "1h"
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

    # Get 4H data for trend direction (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    # Get 1D data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate 4H EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)

    # Calculate 1D Camarilla R1 and S1 levels from previous 1D OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values

    camarilla_r1 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
    camarilla_s1 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12

    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Volume confirmation: current volume > 1.5x average of last 10 periods
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ok[i]) or
            np.isnan(session_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Only trade during session
        if not session_ok[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: bullish if price > 4H EMA50, bearish if price < 4H EMA50
        bullish_trend = close[i] > ema_4h_aligned[i]
        bearish_trend = close[i] < ema_4h_aligned[i]

        if position == 0:
            # LONG: Break above Camarilla R1 with bullish trend and volume confirmation
            if (close[i] > camarilla_r1_aligned[i] and bullish_trend and volume_ok[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Break below Camarilla S1 with bearish trend and volume confirmation
            elif (close[i] < camarilla_s1_aligned[i] and bearish_trend and volume_ok[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or trend turns bearish
            if close[i] < camarilla_r1_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or trend turns bullish
            if close[i] > camarilla_s1_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals