#!/usr/bin/env python3
# 4h_TRIX_Signal_12hTrend_VolumeSpike
# Hypothesis: TRIX (1-period rate of change of triple EMA) generates momentum signals.
# Long when TRIX crosses above zero with 12h uptrend and volume spike; short when crosses below zero with 12h downtrend and volume spike.
# Uses volume confirmation and trend filter to reduce false signals. Designed for 4h to balance trade frequency and accuracy.
# Works in bull/bear markets: TRIX captures momentum shifts; trend filter ensures alignment with higher timeframe.

name = "4h_TRIX_Signal_12hTrend_VolumeSpike"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Calculate TRIX: 1-period ROC of triple EMA (15,15,15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # ROC of triple EMA: (current - previous) / previous * 100
    trix = np.zeros_like(close)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # First value remains 0 (no prior value)

    # Volume confirmation: current volume > 2.0x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(trix[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]

        if position == 0:
            # LONG: TRIX crosses above zero in uptrend with volume spike
            if trix[i] > 0 and trix[i-1] <= 0 and uptrend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero in downtrend with volume spike
            elif trix[i] < 0 and trix[i-1] >= 0 and downtrend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend reversal
            if trix[i] < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend reversal
            if trix[i] > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals