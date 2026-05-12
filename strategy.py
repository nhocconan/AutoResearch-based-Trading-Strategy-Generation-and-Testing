#!/usr/bin/env python3

# 6h_1W_Camarilla_R3S3_Breakout_Trend
# Hypothesis: Long when price breaks above weekly R3 with bullish 1d trend (EMA34 > EMA89), short when breaks below weekly S3 with bearish 1d trend.
# Weekly R3/S3 act as strong support/resistance; 1d EMA crossover filters for trend alignment. Low frequency to avoid fee drag.

name = "6h_1W_Camarilla_R3S3_Breakout_Trend"
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

    # Get weekly data for Camarilla levels (based on previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly high, low, close for Camarilla
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Previous week's OHLC for current week's Camarilla levels
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    prev_close_1w[0] = close_1w[0]

    rang_1w = prev_high_1w - prev_low_1w
    R3 = prev_close_1w + rang_1w * 1.1 / 4
    S3 = prev_close_1w - rang_1w * 1.1 / 4

    # Get daily data for trend filter (EMA34 > EMA89 = bullish, < = bearish)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)

    bullish_trend = ema34_1d_aligned > ema89_1d_aligned
    bearish_trend = ema34_1d_aligned < ema89_1d_aligned

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(bullish_trend[i]) or
            np.isnan(bearish_trend[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above weekly R3 with bullish 1d trend
            if close[i] > R3[i] and close[i-1] <= R3[i-1] and bullish_trend[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below weekly S3 with bearish 1d trend
            elif close[i] < S3[i] and close[i-1] >= S3[i-1] and bearish_trend[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly S3 or trend turns bearish
            if close[i] < S3[i] and close[i-1] >= S3[i-1] or not bullish_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly R3 or trend turns bullish
            if close[i] > R3[i] and close[i-1] <= R3[i-1] or not bearish_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals