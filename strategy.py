#!/usr/bin/env python3
# 4h_TRIX_9_VolumeSpike_ChopFilter
# Hypothesis: TRIX(9) captures momentum reversals in both bull and bear markets.
# Combines with volume spike and Choppiness Index regime filter to avoid false signals.
# TRIX > 0 and rising indicates bullish momentum; TRIX < 0 and falling indicates bearish momentum.
# Volume spike confirms institutional participation.
# Choppiness Index > 61.8 indicates ranging market (avoid trend signals), < 38.2 indicates trending (follow TRIX).
# Targets 20-40 trades/year on 4h timeframe to minimize fee drag.

name = "4h_TRIX_9_VolumeSpike_ChopFilter"
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

    # Get daily data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate TRIX (9-period)
    # TRIX = EMA(EMA(EMA(close, 9), 9), 9) - then % change
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = pd.Series(ema3).pct_change(periods=1).values * 100  # percentage

    # Calculate Choppiness Index (14-period) on daily data
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,14) - min(low,14))) / log10(14)
    atr_1d = np.zeros(len(high_1d))
    for i in range(1, len(high_1d)):
        tr = max(high_1d[i] - low_1d[i],
                 abs(high_1d[i] - close_1d[i-1]),
                 abs(low_1d[i] - close_1d[i-1]))
        atr_1d[i] = tr
    # Smooth ATR with SMA 14
    atr_ma_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_ma_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_14 - min_low_14
    # Avoid division by zero
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_raw = 100 * np.log10(sum_atr_14 / chop_denom) / np.log10(14)
    chop_1d = chop_raw  # Already in 0-100 range

    # Align TRIX and Chop to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)

    # Volume spike: volume > 2.5 * 20-period average (~10 days at 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX positive and rising + volume spike + trending market (CHOP < 38.2)
            if trix_aligned[i] > 0 and trix_aligned[i] > trix_aligned[i-1] and volume_spike[i] and chop_aligned[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX negative and falling + volume spike + trending market (CHOP < 38.2)
            elif trix_aligned[i] < 0 and trix_aligned[i] < trix_aligned[i-1] and volume_spike[i] and chop_aligned[i] < 38.2:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns negative or chop becomes too high (ranging)
            if trix_aligned[i] < 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns positive or chop becomes too high (ranging)
            if trix_aligned[i] > 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals