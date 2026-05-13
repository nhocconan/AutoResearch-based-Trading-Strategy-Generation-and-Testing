#!/usr/bin/env python3
# 4h_TRIX_ZeroLag_VolumeSpike_Direction
# Hypothesis: TRIX (triple exponential average) on 4h with zero-lag crossovers + volume spike confirmation
# to filter false signals. Works in both bull/bear regimes by capturing momentum shifts with
# statistical significance (avoiding whipsaw). Target: 25-40 trades/year per symbol.

name = "4h_TRIX_ZeroLag_VolumeSpike_Direction"
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

    # Get 4h data for TRIX calculation
    df_4h = get_htf_data(prices, '4h')

    # Calculate TRIX (15-period triple EMA, then 1-period ROC)
    ema1 = pd.Series(df_4h['close'].values).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = (ema3 / ema3.shift(1) - 1) * 100  # Percentage change
    trix = trix_raw.values

    # Zero-lag TRIX: TRIX + (TRIX - delayed TRIX) to reduce lag
    trix_delayed = np.roll(trix, 1)
    trix_delayed[0] = 0
    trix_zl = trix + (trix - trix_delayed)
    trix_zl = np.where(np.isnan(trix_zl), 0, trix_zl)  # Handle NaN from roll

    # Align zero-lag TRIX to lower timeframe (4h -> 4h is direct, but using for consistency)
    trix_zl_aligned = align_htf_to_ltf(prices, df_4h, trix_zl)

    # Volume confirmation: current volume > 1.8 x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after sufficient warmup for TRIX
        # Skip if any required value is NaN
        if (np.isnan(trix_zl_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Zero-lag TRIX crosses above zero with volume spike
            if (trix_zl_aligned[i] > 0 and 
                trix_zl_aligned[i-1] <= 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Zero-lag TRIX crosses below zero with volume spike
            elif (trix_zl_aligned[i] < 0 and 
                  trix_zl_aligned[i-1] >= 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Zero-lag TRIX crosses below zero
            if trix_zl_aligned[i] < 0 and trix_zl_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Zero-lag TRIX crosses above zero
            if trix_zl_aligned[i] > 0 and trix_zl_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals