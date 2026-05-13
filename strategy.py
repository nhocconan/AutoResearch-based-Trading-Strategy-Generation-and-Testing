#!/usr/bin/env python3
# 1d_Trix_13_Trend_Filter_Volume_Spike
# Hypothesis: Use TRIX(13) for momentum with 1w EMA13 trend filter and volume spike confirmation.
# Long when TRIX crosses above zero and price is above weekly EMA13 and volume spikes.
# Short when TRIX crosses below zero and price is below weekly EMA13 and volume spikes.
# Exit when TRIX crosses back through zero or volume drops.
# Works in bull markets (momentum continuation) and bear markets (counter-trend spikes during panic/reversal).
# Low frequency due to strict TRIX crossover and volume confirmation requirements.

name = "1d_Trix_13_Trend_Filter_Volume_Spike"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values

    # TRIX(13): triple-smoothed EMA of log returns
    # Calculate once on daily close
    roc = np.diff(np.log(close), prepend=np.log(close[0]))  # log returns
    ema1 = pd.Series(roc).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema2 = pd.Series(ema1).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema3 = pd.Series(ema2).ewm(span=13, adjust=False, min_periods=13).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)  # percentage change
    trix[0] = 0  # first value undefined

    # Weekly EMA13 for trend filter
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Volume spike: volume > 2.0 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    # Align weekly EMA13 to daily
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(ema13_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + weekly uptrend + volume spike
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema13_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + weekly downtrend + volume spike
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema13_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR volume drops
            if trix[i] < 0 and trix[i-1] >= 0 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR volume drops
            if trix[i] > 0 and trix[i-1] <= 0 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals