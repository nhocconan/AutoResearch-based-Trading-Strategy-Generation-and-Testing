#!/usr/bin/env python3
# 12h_Trix_Signal_1dTrend_Volume
# Hypothesis: Use TRIX (15) crossing zero for momentum signals with 1d EMA50 trend filter and volume confirmation.
# Long when TRIX crosses above zero in uptrend with volume spike, short when TRIX crosses below zero in downtrend with volume spike.
# Exit when TRIX reverses or trend changes.
# TRIX is a momentum oscillator that filters out minor cycles, effective in both trending and ranging markets.
# Designed for low trade frequency (20-50 total trades over 4 years) with clear entry/exit rules to avoid overtrading.

name = "12h_Trix_Signal_1dTrend_Volume"
timeframe = "12h"
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

    # Get 1d data for TRIX calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate TRIX (15): Triple EMA of ROC
    # ROC = (close / close.shift(1) - 1) * 100
    roc = np.diff(np.log(close_pd)) * 100 if len(close_pd := pd.Series(df_1d['close'])) > 1 else np.array([])
    if len(roc) < 1:
        ema1 = ema2 = ema3 = np.array([])
    else:
        ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
        ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
        ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Pad to match df_1d length
    trix_raw = np.full_like(df_1d['close'], np.nan, dtype=float)
    if len(ema3) > 0:
        trix_raw[14:] = ema3  # TRIX starts at index 14 due to 3x EMA(15)
    
    # Align TRIX to 12h timeframe
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_raw)

    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    prev_trix = np.nan

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + price above 1d EMA50 (uptrend) + volume spike
            if (prev_trix <= 0 and trix_1d_aligned[i] > 0 and 
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + price below 1d EMA50 (downtrend) + volume spike
            elif (prev_trix >= 0 and trix_1d_aligned[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend changes (price below EMA50)
            if (trix_1d_aligned[i] < 0 or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend changes (price above EMA50)
            if (trix_1d_aligned[i] > 0 or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

        prev_trix = trix_1d_aligned[i]

    return signals