#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Camarilla R3/S3 breakouts with 1-week trend filter and volume confirmation
capture institutional levels and work in both bull and bear markets.
- Camarilla levels act as institutional support/resistance
- 1-week EMA filter ensures trading with higher timeframe trend
- Volume confirmation reduces false breakouts
- Designed for ~15-25 trades/year to minimize fee drag on 12h timeframe
"""

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate Camarilla levels from previous day
    # Using daily OHLC from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align to 12h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)

    # Calculate Camarilla levels
    # R3 = Close + 1.1 * (High - Low) * 1.1/2
    # S3 = Close - 1.1 * (High - Low) * 1.1/2
    camarilla_range = prev_high_aligned - prev_low_aligned
    camarilla_r3 = prev_close_aligned + camarilla_range * 1.1 * 1.1 / 2
    camarilla_s3 = prev_close_aligned - camarilla_range * 1.1 * 1.1 / 2

    # Calculate ATR for stoploss (20-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values

    # Get 1-week EMA21 for trend filter
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)

    # Volume confirmation: 2.0x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema21_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R3 + volume + 1w uptrend
            if (close[i] > camarilla_r3[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 + volume + 1w downtrend
            elif (close[i] < camarilla_s3[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla R3 OR 1w trend turns down
            if close[i] < camarilla_r3[i] or close[i] < ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla S3 OR 1w trend turns up
            if close[i] > camarilla_s3[i] or close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals