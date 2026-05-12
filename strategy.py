#!/usr/bin/env python3
# 1h_4h_1D_Camarilla_R3S3_Breakout_Trend_Volume
# Hypothesis: On 1h timeframe, use 4h trend (EMA50) and 1d Camarilla R3/S3 levels for direction, with volume spike confirmation.
# Enter long when price breaks above 1d R3 with 4h bullish trend and volume spike.
# Enter short when price breaks below 1d S3 with 4h bearish trend and volume spike.
# Uses 4h/1d for signal direction, 1h only for entry timing. Targets 15-35 trades/year to avoid fee drag.

name = "1h_4h_1D_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "1h"
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

    # Get 4h data for trend (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    # Calculate EMA50 on 4h close
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)

    # Get 1d data for Camarilla R3/S3 levels (from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate Camarilla levels from previous 1d OHLC (to avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4

    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # Volume confirmation: current volume > 2.0x average of last 24 periods (1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    # Session filter: 08-20 UTC (already datetime64[ms], use .hour on index)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ok[i]) or
            np.isnan(session_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend from 4h EMA50: price above/below EMA
        bullish_trend = close[i] > ema_4h_aligned[i]
        bearish_trend = close[i] < ema_4h_aligned[i]

        if position == 0:
            # LONG: Break above 1d R3 with bullish 4h trend and volume spike during session
            if (close[i] > camarilla_r3_aligned[i] and bullish_trend and volume_ok[i] and session_ok[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Break below 1d S3 with bearish 4h trend and volume spike during session
            elif (close[i] < camarilla_s3_aligned[i] and bearish_trend and volume_ok[i] and session_ok[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend turns bearish
            if close[i] < camarilla_r3_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend turns bullish
            if close[i] > camarilla_s3_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals