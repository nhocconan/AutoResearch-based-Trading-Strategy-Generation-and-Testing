#!/usr/bin/env python3
# 6h_TRIX_ZeroLag_VolumeSpike_Direction
# Hypothesis: TRIX (triple exponential average) zero-lag version on 6h with volume spike and direction filter.
# Zero-lag TRIX reduces lag by adding momentum component. Trades only when TRIX crosses zero with volume confirmation.
# Direction filter: 1w EMA200 to ensure alignment with long-term trend.
# Volume spike: current volume > 1.5x 50-period average to filter low-quality breakouts.
# Designed to capture momentum shifts in both bull and bear markets with trend alignment.
# Target: 12-37 trades/year per symbol to minimize fee drag while maintaining edge.

name = "6h_TRIX_ZeroLag_VolumeSpike_Direction"
timeframe = "6h"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # Calculate zero-lag TRIX on 6h
    # EMA1 = EMA(close, 12)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2 = EMA(EMA1, 12)
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3 = EMA(EMA2, 12)
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = (EMA3 - prev_EMA3) / prev_EMA3 * 100
    trix = np.zeros_like(close)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # Zero-lag TRIX = 2*TRIX - lagged TRIX (1 period)
    trix_lag = np.roll(trix, 1)
    trix_lag[0] = 0
    trix_zero_lag = 2 * trix - trix_lag

    # Trend filter: 1w EMA200
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)

    # Volume confirmation: current volume > 1.5 x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(trix_zero_lag[i]) or 
            np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Zero-lag TRIX crosses above zero with volume spike and above weekly EMA200
            if (trix_zero_lag[i] > 0 and trix_zero_lag[i-1] <= 0 and 
                close[i] > ema200_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Zero-lag TRIX crosses below zero with volume spike and below weekly EMA200
            elif (trix_zero_lag[i] < 0 and trix_zero_lag[i-1] >= 0 and 
                  close[i] < ema200_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Zero-lag TRIX crosses below zero
            if trix_zero_lag[i] < 0 and trix_zero_lag[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Zero-lag TRIX crosses above zero
            if trix_zero_lag[i] > 0 and trix_zero_lag[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals