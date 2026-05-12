#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Reversal_v1
Hypothesis: In BTC/ETH mean-reverting markets (especially during 2025 bear/range), price reacts at Camarilla pivot levels (S1/S3/R1/R3) calculated from prior 1d range. Enter mean-reversion trades when price touches these levels with volume confirmation and only when market is in chop regime (Chop > 61.8). Exit on opposite touch or trend resumption. Uses 1d for pivots and regime filter, 4h for execution. Targets low trade frequency to avoid fee drag while capturing reversals in both bull and bear markets.
"""

name = "4h_Camarilla_Pivot_Reversal_v1"
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

    # Get 1d data for Camarilla pivots and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels for each 1d bar
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We'll use S1, S3, R1, R3 as key reversal levels
    rng_1d = high_1d - low_1d
    r1_1d = close_1d + rng_1d * 1.1 / 12
    r3_1d = close_1d + rng_1d * 1.1 / 4
    s1_1d = close_1d - rng_1d * 1.1 / 12
    s3_1d = close_1d - rng_1d * 1.1 / 4

    # Align Camarilla levels to 4h (1d levels are valid for the entire day after close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)

    # Chop regime filter: Chop > 61.8 = ranging market (good for mean reversion)
    # Calculate Chop on 1d: CHOP = 100 * log10(SUM(ATR(1), n) / (n * MAX(H-L, n))) / log10(n)
    # Simplified: use rolling ATR and range
    atr_1d = np.zeros(len(high_1d))
    tr_1d = np.zeros(len(high_1d))
    for i in range(len(high_1d)):
        if i == 0:
            tr_1d[i] = high_1d[i] - low_1d[i]
        else:
            tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        if i < 1:
            atr_1d[i] = tr_1d[i]
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14  # Wilder's smoothing
    
    # Avoid division by zero
    max_range_1d = np.maximum(high_1d - low_1d, 1e-10)
    sum_atr_14 = np.zeros_like(atr_1d)
    sum_range_14 = np.zeros_like(max_range_1d)
    for i in range(len(atr_1d)):
        if i < 14:
            sum_atr_14[i] = np.sum(atr_1d[max(0, i-13):i+1])
            sum_range_14[i] = np.sum(max_range_1d[max(0, i-13):i+1])
        else:
            sum_atr_14[i] = sum_atr_14[i-1] - atr_1d[i-14] + atr_1d[i]
            sum_range_14[i] = sum_range_14[i-1] - max_range_1d[i-14] + max_range_1d[i]
    
    chop_1d = 100 * np.log10(sum_atr_14 / sum_range_14) / np.log10(14)
    chop_1d = np.where(np.isnan(chop_1d) | np.isinf(chop_1d), 50, chop_1d)  # neutral if undefined
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)

    # Volume confirmation: volume > 1.5x 24-period average on 4h
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Only trade in chop regime (chop > 61.8 = ranging)
        if chop_1d_aligned[i] <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches S1 or S3 with volume confirmation
            if (low[i] <= s1_1d_aligned[i] or low[i] <= s3_1d_aligned[i]) and volume[i] > vol_avg_24[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches R1 or R3 with volume confirmation
            elif (high[i] >= r1_1d_aligned[i] or high[i] >= r3_1d_aligned[i]) and volume[i] > vol_avg_24[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches R1 or R3 (opposite side) or chop breaks down
            if (high[i] >= r1_1d_aligned[i] or high[i] >= r3_1d_aligned[i]) or chop_1d_aligned[i] < 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches S1 or S3 (opposite side) or chop breaks down
            if (low[i] <= s1_1d_aligned[i] or low[i] <= s3_1d_aligned[i]) or chop_1d_aligned[i] < 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals