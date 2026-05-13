#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_ChopRegime
# Hypothesis: TRIX (Triple Exponential Average) crossing above/below zero with momentum, combined with volume spike and chop regime filter, captures trend changes while avoiding whipsaws in range markets. Works in bull (TRIX up + volume + low chop) and bear (TRIX down + volume + low chop) by following momentum with confirmation. Target: 20-40 trades/year per symbol.

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # TRIX: 15-period triple EMA of ROC
    close_series = pd.Series(close)
    roc = close_series.pct_change(periods=1)  # 1-period rate of change
    ema1 = roc.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3 * 10000).values  # scale for readability

    # Chop regime: Chop index (14) < 38.2 = trending regime (use TRIX signals)
    # Chop = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first bar has no prior close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(tr) / (hh - ll + 1e-10)) / np.log10(14)  # avoid div0
    chop[np.isnan(chop)] = 100  # neutral if undefined
    chop_regime = chop < 38.2  # trending regime

    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(chop_regime[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + volume spike + trending regime
            if trix[i] > 0 and trix[i-1] <= 0 and volume_spike[i] and chop_regime[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + volume spike + trending regime
            elif trix[i] < 0 and trix[i-1] >= 0 and volume_spike[i] and chop_regime[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or chop regime ends (range)
            if trix[i] < 0 and trix[i-1] >= 0 or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or chop regime ends (range)
            if trix[i] > 0 and trix[i-1] <= 0 or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals