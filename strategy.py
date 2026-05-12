#!/usr/bin/env python3
# 12h_1D_1W_Camarilla_R3S3_Breakout_Volume_Trend
# Hypothesis: 12-hour breakouts above daily R3 or below daily S3 with volume confirmation and weekly trend filter.
# Uses daily timeframe for trend and pivot levels, weekly for context, to reduce noise and avoid overtrading.
# Designed for 12-37 trades/year. Weekly trend filter avoids whipsaws in range markets; volume confirms institutional interest.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by following the weekly trend.

name = "12h_1D_1W_Camarilla_R3S3_Breakout_Volume_Trend"
timeframe = "12h"
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
    volume = prices['volume'].values

    # Get daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Get weekly data for context (trend confirmation)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate daily Camarilla levels (R3 and S3) based on previous day
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    rang = prev_high - prev_low
    R3 = prev_close + rang * 1.1 / 4
    S3 = prev_close - rang * 1.1 / 4

    # Align daily levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)

    # Weekly trend filter: price above/below weekly EMA
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    weekly_bullish = close > ema_1w_aligned
    weekly_bearish = close < ema_1w_aligned

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_ok[i]) or np.isnan(weekly_bullish[i]) or np.isnan(weekly_bearish[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Daily trend filter
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Price closes above R3 with bullish daily trend, weekly confirmation, and volume
            if (close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1] and 
                bullish_trend and weekly_bullish[i] and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S3 with bearish daily trend, weekly confirmation, and volume
            elif (close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1] and 
                  bearish_trend and weekly_bearish[i] and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S3 or daily trend turns bearish
            if (close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1]) or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R3 or daily trend turns bullish
            if (close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1]) or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals