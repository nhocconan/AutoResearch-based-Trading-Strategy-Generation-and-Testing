#!/usr/bin/env python3
"""
1d_TRIX_0_Long_Short_1wTrend_VolumeSpike
Hypothesis: TRIX(12) crossing zero with 1w EMA50 trend filter and volume confirmation (1.5x average) captures momentum shifts in both bull and bear markets. TRIX zero-cross filters noise, and 1w trend ensures alignment with higher timeframe momentum. Volume spike confirms conviction. Works in ranging markets via trend filter and in trending markets via momentum capture.
"""

name = "1d_TRIX_0_Long_Short_1wTrend_VolumeSpike"
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

    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate TRIX(12): triple EMA of ROC, then % change
    # TRIX = 100 * (EMA3(EMA2(EMA1(ROC))) - previous) / previous
    # Simplified: TRIX ≈ 100 * (EMA3(close) - EMA3(close)_prev) / EMA3(close)_prev
    # We'll use zero-cross of smoothed momentum
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.diff() / ema3.shift(1)).values
    trix = np.where(np.isnan(trix), 0, trix)  # handle initial NaN

    # Align TRIX to 1d (already same tf, but for consistency)
    # Actually, TRIX is calculated on close, so no alignment needed
    trix_signal = trix

    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume spike: >1.5x 20-period average (1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + 1w EMA50 uptrend + volume spike
            if (trix_signal[i] > 0 and trix_signal[i-1] <= 0 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + 1w EMA50 downtrend + volume spike
            elif (trix_signal[i] < 0 and trix_signal[i-1] >= 0 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero
            if trix_signal[i] < 0 and trix_signal[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero
            if trix_signal[i] > 0 and trix_signal[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals