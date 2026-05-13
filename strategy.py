#!/usr/bin/env python3
# 4h_TRIX_ZeroLag_VolumeSpike_TrendFilter
# Hypothesis: TRIX zero-lag (Ehlers) identifies momentum shifts early, combined with volume spike and 1-day EMA trend filter for confirmation.
# Works in bull/bear: long when Trix crosses above signal line with volume and above daily EMA; short when crosses below with volume and below daily EMA.
# Designed for 20-40 trades/year to minimize fee drag.

name = "4h_TRIX_ZeroLag_VolumeSpike_TrendFilter"
timeframe = "4h"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')

    # 1-day EMA34 trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # TRIX zero-lag (Ehlers) on close
    # EMA1
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    # EMA2
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    # EMA3
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    # TRIX = 100 * (EMA3 - EMA3_prev) / EMA3_prev
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values
    # Signal line: EMA of TRIX
    signal_line = pd.Series(trix).ewm(span=8, adjust=False, min_periods=8).mean().values

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(12, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(signal_line[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above signal line with volume spike and above daily EMA34
            if (trix[i] > signal_line[i] and trix[i-1] <= signal_line[i-1] and 
                volume_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below signal line with volume spike and below daily EMA34
            elif (trix[i] < signal_line[i] and trix[i-1] >= signal_line[i-1] and 
                  volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below signal line or closes below daily EMA34
            if (trix[i] < signal_line[i] and trix[i-1] >= signal_line[i-1]) or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above signal line or closes above daily EMA34
            if (trix[i] > signal_line[i] and trix[i-1] <= signal_line[i-1]) or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals