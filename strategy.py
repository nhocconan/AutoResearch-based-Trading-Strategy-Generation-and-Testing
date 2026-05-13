#!/usr/bin/env python3
# 1d_TRIX_ZeroCross_VolumeFilter
# Hypothesis: TRIX zero-cross on daily timeframe with volume confirmation and 1-week trend filter.
# Long when TRIX crosses above zero with rising volume and 1-week uptrend.
# Short when TRIX crosses below zero with rising volume and 1-week downtrend.
# Exit on opposite TRIX cross or trend reversal.
# Designed to capture momentum shifts in both bull and bear markets with low trade frequency.

name = "1d_TRIX_ZeroCross_VolumeFilter"
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

    # Get 1d data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    
    # TRIX: triple EMA of percentage change, period=15
    close_series = pd.Series(df_1d['close'])
    roc = close_series.pct_change()
    ema1 = roc.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3 * 100  # scale for readability
    trix_values = trix.values

    # Align TRIX to minute timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_values)

    # 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_aligned[i-1]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # TRIX zero-cross detection
        trix_cross_up = trix_aligned[i-1] <= 0 and trix_aligned[i] > 0
        trix_cross_down = trix_aligned[i-1] >= 0 and trix_aligned[i] < 0

        if position == 0:
            # LONG: TRIX crosses up + volume spike + 1-week uptrend
            if trix_cross_up and volume[i] > vol_avg_20[i] * 1.5:
                if close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: TRIX crosses down + volume spike + 1-week downtrend
            elif trix_cross_down and volume[i] > vol_avg_20[i] * 1.5:
                if close[i] < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses down or trend turns down
            if trix_cross_down:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_50_1w_aligned[i]:  # trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses up or trend turns up
            if trix_cross_up:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_50_1w_aligned[i]:  # trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals