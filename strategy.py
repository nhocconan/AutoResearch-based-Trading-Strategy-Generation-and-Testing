#!/usr/bin/env python3
# 1h_TRIX_ZeroLag_VolumeSpike_Direction
# Hypothesis: TRIX zero-lag crossover with volume spike and 4h/1d trend alignment.
# Uses zero-lag TRIX to reduce lag in trend detection, volume surge to confirm institutional interest,
# and higher timeframe trends (4h/1d) to filter counter-trend trades. Designed for 1h timeframe
# with tight entry conditions (target: 15-35 trades/year) to avoid fee drag. Works in bull/bear
# by following dominant higher timeframe trend.

name = "1h_TRIX_ZeroLag_VolumeSpike_Direction"
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

    # Calculate zero-lag TRIX (15-period)
    # Step 1: EMA1
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Step 2: EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Step 3: EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Step 4: TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix_raw = np.full(n, np.nan)
    trix_raw[15:] = (ema3[15:] - ema3[14:-1]) / ema3[14:-1] * 100
    # Step 5: Signal line (EMA of TRIX, 9-period)
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    # Step 6: Zero-lag TRIX = 2*TRIX - signal
    zl_trix = 2 * trix_raw - trix_signal

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if data is not ready
        if (np.isnan(zl_trix[i]) or np.isnan(zl_trix[i-1]) or np.isnan(trix_signal[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Check session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: zero-lag TRIX crosses above signal line with volume spike and 4h/1d uptrend
            if (zl_trix[i] > trix_signal[i] and zl_trix[i-1] <= trix_signal[i-1] and
                volume_spike[i] and close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: zero-lag TRIX crosses below signal line with volume spike and 4h/1d downtrend
            elif (zl_trix[i] < trix_signal[i] and zl_trix[i-1] >= trix_signal[i-1] and
                  volume_spike[i] and close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: zero-lag TRIX crosses below signal line
            if zl_trix[i] < trix_signal[i] and zl_trix[i-1] >= trix_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: zero-lag TRIX crosses above signal line
            if zl_trix[i] > trix_signal[i] and zl_trix[i-1] <= trix_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals