#!/usr/bin/env python3
# 6h_TRIX_VolumeSpike_1dTrend
# Hypothesis: Use TRIX (triple EMA) on 6h for momentum, confirmed by volume spikes.
# Long when TRIX crosses above zero with volume spike and 1d EMA34 uptrend.
# Short when TRIX crosses below zero with volume spike and 1d EMA34 downtrend.
# Exit when TRIX returns to zero.
# Designed to work in both bull (buy momentum in uptrend) and bear (sell momentum in downtrend).
# Target: 15-30 trades/year per symbol.

name = "6h_TRIX_VolumeSpike_1dTrend"
timeframe = "6h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate TRIX on 6h: triple EMA of percent change
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.pct_change())
    trix = trix.fillna(0).values

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        # Skip if data is not ready
        if np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero with volume spike and 1d EMA uptrend
            if trix[i] > 0 and trix[i-1] <= 0 and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero with volume spike and 1d EMA downtrend
            elif trix[i] < 0 and trix[i-1] >= 0 and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX returns to zero
            if trix[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX returns to zero
            if trix[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals