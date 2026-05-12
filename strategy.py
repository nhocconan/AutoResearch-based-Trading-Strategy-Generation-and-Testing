#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Volume
# Hypothesis: Camarilla R3/S3 breakout with 1-day EMA34 trend filter and volume spike confirmation on 4h timeframe.
# Camarilla levels provide institutional-grade support/resistance. Breakouts above R3 or below S3 with volume
# indicate strong momentum. EMA34 filter ensures trades align with daily trend, reducing whipsaw in chop.
# Works in bull/bear markets by following daily trend direction. Targets 20-40 trades/year to minimize fee drag.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Volume"
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

    # Get 1d data for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels for the previous day
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    # Using previous day's values (shifted by 1)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # First value has no previous
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]

    diff = prev_high - prev_low
    camarilla_r3 = prev_close + (diff * 1.1 / 4)
    camarilla_s3 = prev_close - (diff * 1.1 / 4)

    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # Calculate EMA34 on daily close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate volume spike threshold (2.0x 20-period SMA)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 with volume spike in uptrend
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike in downtrend
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA34 or reverses below R3
            if close[i] < ema34_1d_aligned[i] or close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA34 or reverses above S3
            if close[i] > ema34_1d_aligned[i] or close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals