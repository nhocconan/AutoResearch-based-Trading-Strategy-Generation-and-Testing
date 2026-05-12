#!/usr/bin/env python3
# 12h_1W_1D_Camarilla_R3S3_Breakout_Volume_Trend
# Hypothesis: Breakouts above weekly R3 or below weekly S3 on 12h timeframe with volume confirmation and 1d EMA34 trend filter. Uses weekly for structure (reduces noise) and daily for trend filter. Designed for 12-37 trades/year to minimize fee drift while capturing strong momentum moves in both bull and bear markets.

name = "12h_1W_1D_Camarilla_R3S3_Breakout_Volume_Trend"
timeframe = "12h"
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

    # Get weekly data for structure (Camarilla levels)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate weekly Camarilla levels (R3 and S3) based on previous week
    prev_high = np.roll(df_1w['high'].values, 1)
    prev_low = np.roll(df_1w['low'].values, 1)
    prev_close = np.roll(df_1w['close'].values, 1)
    prev_high[0] = df_1w['high'].values[0]
    prev_low[0] = df_1w['low'].values[0]
    prev_close[0] = df_1w['close'].values[0]
    
    rang = prev_high - prev_low
    R3 = prev_close + rang * 1.1 / 4
    S3 = prev_close - rang * 1.1 / 4

    # Align weekly levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_ok[i])):
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
            # LONG: Price crosses above weekly R3 with bullish daily trend and volume confirmation
            if close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below weekly S3 with bearish daily trend and volume confirmation
            elif close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly S3 or daily trend turns bearish
            if close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly R3 or daily trend turns bullish
            if close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals