#!/usr/bin/env python3
# 4h_1D_1W_Camarilla_R1S1_Breakout_Trend_Volume
# Hypothesis: Enter long when price breaks above daily Camarilla R1 with bullish weekly trend and volume confirmation; enter short when price breaks below daily S1 with bearish weekly trend and volume confirmation. Exit on opposite breakout or trend reversal. Uses R1/S1 for tighter entries than R3/S3, aiming for 20-50 trades/year. Weekly trend filter ensures alignment with higher timeframe momentum to avoid counter-trend trades. Volume confirmation filters false breakouts. Designed for low frequency to minimize fee drag in both bull and bear markets.

name = "4h_1D_1W_Camarilla_R1S1_Breakout_Trend_Volume"
timeframe = "4h"
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

    # Calculate daily Camarilla levels (R1 and S1) based on previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    rang = prev_high - prev_low
    R1 = prev_close + rang * 1.1 / 12
    S1 = prev_close - rang * 1.1 / 12

    # Align daily levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
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
            # LONG: Price crosses above R1 with bullish weekly trend and volume confirmation
            if close[i] > R1_aligned[i] and close[i-1] <= R1_aligned[i-1] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below S1 with bearish weekly trend and volume confirmation
            elif close[i] < S1_aligned[i] and close[i-1] >= S1_aligned[i-1] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 or weekly trend turns bearish
            if close[i] < S1_aligned[i] and close[i-1] >= S1_aligned[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 or weekly trend turns bullish
            if close[i] > R1_aligned[i] and close[i-1] <= R1_aligned[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals