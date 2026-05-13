#!/usr/bin/env python3
# 4h_TRIX_ZeroLag_VolumeSpike_TrendFilter_v1
# Hypothesis: TRIX zero-crossings with volume confirmation and 12h EMA50 trend filter on 4h timeframe.
# TRIX (triple exponential average) captures momentum shifts; zero-crossings signal trend changes.
# Volume spike confirms institutional participation. 12h EMA50 filter ensures alignment with higher timeframe trend.
# Exit when TRIX re-crosses zero or volume drops below average to avoid whipsaws.
# Designed for 20-40 trades/year to minimize fee drag. Works in bull/bear by capturing momentum shifts with trend alignment.

name = "4h_TRIX_ZeroLag_VolumeSpike_TrendFilter_v1"
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
    volume = prices['volume'].values

    # TRIX (15-period standard)
    # EMA1 = EMA(close, 15)
    # EMA2 = EMA(EMA1, 15)
    # EMA3 = EMA(EMA2, 15)
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.full(n, np.nan)
    trix[15:] = (ema3[15:] - ema3[14:-1]) / ema3[14:-1] * 100

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if np.isnan(trix[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero with volume spike and price above 12h EMA50 (uptrend)
            if trix[i] > 0 and trix[i-1] <= 0 and volume_spike[i] and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero with volume spike and price below 12h EMA50 (downtrend)
            elif trix[i] < 0 and trix[i-1] >= 0 and volume_spike[i] and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR volume drops below average
            if trix[i] < 0 and trix[i-1] >= 0 or volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR volume drops below average
            if trix[i] > 0 and trix[i-1] <= 0 or volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals