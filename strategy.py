#!/usr/bin/env python3
# 4h_TRIX_ZeroLag_Volume_Spike_1dTrend
# Hypothesis: TRIX with zero-lag smoothing detects momentum shifts early. Long when TRIX crosses above zero with volume spike and daily uptrend.
# Short when TRIX crosses below zero with volume spike and daily downtrend. Uses volume > 1.5x 20-period average for confirmation.
# Designed for 4h timeframe to balance signal quality and trade frequency. Works in bull markets via momentum and bear via mean reversion spikes.

name = "4h_TRIX_ZeroLag_Volume_Spike_1dTrend"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Calculate TRIX (15,9,9) - zero-lag version
    # EMA1: 15-period EMA of close
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2: 15-period EMA of EMA1
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3: 15-period EMA of EMA2
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX: 1-period % change of EMA3
    trix_raw = np.zeros_like(close)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # Signal line: 9-period EMA of TRIX
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    # Zero-lag TRIX: 2*TRIX - signal line (reduces lag)
    trix = 2 * trix_raw - trix_signal

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA34
        price_above_daily_ema = close[i] > ema_34_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_34_1d_aligned[i]

        if position == 0:
            # LONG: TRIX crosses above zero with volume spike and daily uptrend
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                volume_ok[i] and price_above_daily_ema):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero with volume spike and daily downtrend
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  volume_ok[i] and price_below_daily_ema):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or momentum fades
            if trix[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or momentum fades
            if trix[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals