#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot breakout at R3/S3 levels from daily pivots, filtered by weekly trend (EMA34) and volume confirmation.
# Camarilla levels act as intraday support/resistance; breakouts beyond R3/S3 indicate strong momentum.
# Weekly EMA34 filter ensures alignment with higher timeframe trend, reducing counter-trend trades.
# Volume confirmation adds validity to breakout moves.
# Designed for 12h timeframe to target 50-150 total trades over 4 years with controlled frequency.
# Works in bull markets via R3 breakouts and in bear markets via S3 breakdowns.

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

    # Get 1d data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # Calculate Camarilla levels (R3, S3) from previous day's OHLC
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    camarilla_r3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])
    camarilla_s3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)

    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Volume filter: >1.5x 20-period average on 12h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Camarilla R3 + price above weekly EMA34 (bullish trend) + volume spike
            if (close[i] > camarilla_r3_aligned[i] and
                close[i] > ema_34_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Camarilla S3 + price below weekly EMA34 (bearish trend) + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and
                  close[i] < ema_34_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Camarilla S3 or price below weekly EMA34
            if (close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Camarilla R3 or price above weekly EMA34
            if (close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals