#!/usr/bin/env python3
# 1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume
# Hypothesis: Use 4h Camarilla R3/S3 breakout with 4h EMA trend filter and 1d volume spike confirmation, executed on 1h timeframe.
# Camarilla levels provide precise support/resistance; EMA filter ensures trend alignment; volume spike confirms breakout strength.
# Works in bull (buys breakouts above R3 in uptrend) and bear (sells breakdowns below S3 in downtrend).
# Target: 80-150 total trades over 4 years = 20-38/year for 1h.

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for Camarilla calculation and EMA trend
    df_4h = get_htf_data(prices, '4h')
    # Typical price for Camarilla
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3.0
    # Calculate Camarilla levels: R3, S3
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    camarilla_r3 = close_4h + 1.1 * (high_4h - low_4h)
    camarilla_s3 = close_4h - 1.1 * (high_4h - low_4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    # 4h EMA for trend filter (21-period)
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)

    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Camarilla R3 + price above 4h EMA (uptrend) + volume spike
            if (close[i] > camarilla_r3_aligned[i] and
                close[i] > ema_4h_aligned[i] and
                volume[i] > vol_ma_20_1d_aligned[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below Camarilla S3 + price below 4h EMA (downtrend) + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and
                  close[i] < ema_4h_aligned[i] and
                  volume[i] > vol_ma_20_1d_aligned[i] * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Camarilla S3 or 4h EMA turns down
            if (close[i] < camarilla_s3_aligned[i] or close[i] < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price breaks above Camarilla R3 or 4h EMA turns up
            if (close[i] > camarilla_r3_aligned[i] or close[i] > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals