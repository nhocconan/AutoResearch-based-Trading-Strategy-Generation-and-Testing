#!/usr/bin/env python3
# 6h_1D_1W_Camarilla_R3S3_Breakout_Trend_Filter
# Hypothesis: Breakouts at weekly and daily Camarilla R3/S3 levels with trend alignment on 6h.
# Uses 1w for long-term trend bias (EMA50) and 1d for entry levels (R3/S3).
# Requires price to be above/below weekly EMA50 for long/short bias, then breaks daily R3/S3.
# Designed to work in both bull and bear markets by filtering trades with weekly trend.
# Targets 12-37 trades/year on 6h timeframe to avoid fee drag.

name = "6h_1D_1W_Camarilla_R3S3_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values

    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Get 1d data for Camarilla R3/S3 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate Camarilla R3 and S3 from previous 1d OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4

    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price relative to weekly EMA50
        bullish_trend = close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i]

        if position == 0:
            # LONG: Price above weekly EMA50 and breaks above daily R3
            if bullish_trend and close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below weekly EMA50 and breaks below daily S3
            elif bearish_trend and close[i] < camarilla_s3_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly EMA50 or re-enters below R3
            if not bullish_trend or close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly EMA50 or re-enters above S3
            if not bearish_trend or close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals