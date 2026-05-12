#!/usr/bin/env python3
# 160117: 4h_TRIX_VolumeSpike_Regime
# Hypothesis: TRIX (12-period) crossing zero line with volume confirmation (>1.5x 20-bar average) and Choppiness Index regime filter (CHOP > 61.8 for range, < 38.2 for trend) filters whipsaws in sideways markets. TRIX captures momentum shifts, volume confirms conviction, and regime filter ensures trades align with market structure. Works in bull/bear by adapting to regime. Uses 4h timeframe with 1d Choppiness Index for regime context.

name = "4h_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1-day TRIX (12-period)
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then % change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema3).pct_change() * 100  # percentage

    # Calculate 1-day Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (max(high) - min(low)))) / log10(n)
    tr1 = high[1:] - low[:-1]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr14 * 14 / (highest_high - lowest_low)) / np.log10(14)
    chop = chop_raw  # already scaled

    # Align TRIX and Choppiness Index to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + volume confirmation + chop < 38.2 (trending)
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                volume_confirm[i] and chop_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + volume confirmation + chop < 38.2 (trending)
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  volume_confirm[i] and chop_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR chop > 61.8 (range) OR volume drops
            if (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0) or \
               chop_aligned[i] > 61.8 or \
               not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR chop > 61.8 (range) OR volume drops
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0) or \
               chop_aligned[i] > 61.8 or \
               not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals