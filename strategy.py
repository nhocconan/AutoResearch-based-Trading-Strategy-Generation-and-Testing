#!/usr/bin/env python3
# 1h_4h_Camarilla_R3S3_Breakout_Trend_Volume
# Hypothesis: Breakout at 4h Camarilla R3/S3 levels with volume confirmation and 4h trend filter.
# Uses 4h for signal direction/trend and 1h for precise entry timing to reduce whipsaws.
# Targets 15-35 trades/year (~60-140 total over 4 years) to avoid excessive fee drag.
# Works in both bull and bear markets by requiring volume confirmation and trend alignment.

name = "1h_4h_Camarilla_R3S3_Breakout_Trend_Volume"
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

    # Get 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)

    # Calculate Camarilla R3 and S3 levels from previous 4h OHLC
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values

    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4

    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)

    # Volume confirmation: current volume > 1.8x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 4h
        bullish_trend = close[i] > ema_4h_aligned[i]
        bearish_trend = close[i] < ema_4h_aligned[i]

        if position == 0:
            # LONG: Break above Camarilla R3 with bullish trend and volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and bullish_trend and volume_ok[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Break below Camarilla S3 with bearish trend and volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and bearish_trend and volume_ok[i]):
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