#!/usr/bin/env python3
# 1d_TRIX_ZeroLag_VolumeSpike_Direction
# Hypothesis: TRIX (15) with zero-lag smoothing on daily timeframe captures momentum reversals in both bull and bear markets.
# Volume spike (>2x 20-day average) confirms institutional participation.
# Zero-lag TRIX reduces lag for timely signals while maintaining whipsaw resistance.
# Target: 10-25 trades/year per symbol to minimize fee drag while capturing significant moves.

name = "1d_TRIX_ZeroLag_VolumeSpike_Direction"
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

    # Get weekly data for trend filter (optional, can be removed if not needed)
    # For pure 1d strategy, we'll use daily TRIX only

    # Calculate TRIX (15) - triple exponential moving average of percent change
    # TRIX = EMA(EMA(EMA(roc, 15), 15), 15) * 100
    # Where roc = (close - close.shift(1)) / close.shift(1) * 100
    
    # Calculate rate of change
    roc = np.zeros(n)
    roc[1:] = (close[1:] - close[:-1]) / close[:-1] * 100
    
    # First EMA
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Second EMA
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Third EMA (TRIX)
    trix = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values * 100

    # Zero-lag TRIX: TRIX + (TRIX - EMA(TRIX))
    # This reduces lag while preserving turning point signals
    trix_ema = pd.Series(trix).ewm(span=15, adjust=False, min_periods=15).mean().values
    zero_lag_trix = trix + (trix - trix_ema)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after sufficient warmup for TRIX
        # Skip if any required value is NaN
        if (np.isnan(zero_lag_trix[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Zero-lag TRIX crosses above zero with volume spike
            if (zero_lag_trix[i] > 0 and 
                zero_lag_trix[i-1] <= 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Zero-lag TRIX crosses below zero with volume spike
            elif (zero_lag_trix[i] < 0 and 
                  zero_lag_trix[i-1] >= 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Zero-lag TRIX crosses below zero
            if zero_lag_trix[i] < 0 and zero_lag_trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Zero-lag TRIX crosses above zero
            if zero_lag_trix[i] > 0 and zero_lag_trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals