#!/usr/bin/env python3
# 1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeFilter
# Hypothesis: Camarilla pivot R3/S3 breakouts with 4h EMA50 trend filter and volume confirmation on 1h timeframe.
# Camarilla levels identify key intraday support/resistance; breakouts above R3 or below S3 signal strong momentum.
# 4h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume filter confirms institutional participation. Designed for 15-30 trades/year to minimize fee drag.
# Works in bull/bear: long when price breaks above R3 with volume and above 4h EMA50; short when breaks below S3 with volume and below 4h EMA50.

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeFilter"
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

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')

    # Calculate Camarilla pivots from previous day
    # Using previous day's high, low, close
    prev_high = high.copy()
    prev_low = low.copy()
    prev_close = close.copy()
    prev_high[1:] = high[:-1]
    prev_low[1:] = low[:-1]
    prev_close[1:] = close[:-1]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]

    # Daily range
    range_ = prev_high - prev_low
    # Camarilla levels
    R3 = prev_close + range_ * 1.1 / 4
    S3 = prev_close - range_ * 1.1 / 4

    # 4h EMA50 trend filter
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(R3[i]) or 
            np.isnan(S3[i]) or 
            np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 with volume and above 4h EMA50
            if (close[i] > R3[i] and 
                volume_filter[i] and 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3 with volume and below 4h EMA50
            elif (close[i] < S3[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 (reversal) or loses 4h EMA50 support
            if close[i] < S3[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 (reversal) or loses 4h EMA50 resistance
            if close[i] > R3[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals