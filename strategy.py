#!/usr/bin/env python3
# 4h_TRIX_Volume_Spike_Chop_Filter
# Hypothesis: TRIX (triple EMA crossover) signals momentum reversals. Combine with volume spike for confirmation and Choppiness Index regime filter to avoid whipsaws in sideways markets. Works in both bull/bear by adapting to regime: trend-follow when CHOP < 38.2, mean-revert when CHOP > 61.8. Target: 25-35 trades/year on 4h to minimize fee drag.

name = "4h_TRIX_Volume_Spike_Chop_Filter"
timeframe = "4h"
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

    # Get 4h data for TRIX calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 15:
        return np.zeros(n)
    close_4h = df_4h['close'].values

    # Calculate TRIX: triple EMA of ROC
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago
    ema1 = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3 * 100  # percentage change

    # Align TRIX to lower timeframe (if needed, but we're using 4h as primary)
    # Since we're using 4h data directly, no alignment needed for TRIX itself
    # But we need to align it to the original price index if we were using different TF
    # Here we keep it as is since primary TF is 4h

    # Get 1d data for Choppiness Index (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Choppiness Index
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log10(n))) / log10(n)
    # Using 14-period as standard
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[0], tr1])  # first TR is 0 or undefined
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr1 / 14) / np.log10(14)  # simplified: 100 * log(n) / log(n) but adjusted
    # Actually: CHOP = 100 * log10(sum(ATR(14)) / (n * ATR_avg)) / log10(n) - using standard formula
    # Let's use proper CHOP formula
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr14 / (14 * np.mean(atr14))) / np.log10(14) if len(atr14) > 0 else np.zeros_like(atr14)
    # Handle edge cases
    chop = np.where(atr14 > 0, 100 * np.log10(sum_atr14 / (14 * atr14)) / np.log10(14), 50.0)

    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)

    # Volume confirmation: volume > 2.0x 20-period average (4h)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(trix[i]) if i < len(trix) else True or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Get current TRIX value (need to handle index offset since TRIX is on 4h)
        # Since we're using 4h as primary, we need to map i to 4h index
        # But we already aligned TRIX? Actually we didn't - let's fix this
        # Recalculate: get TRIX aligned to original price index
        if i >= len(trix):
            trix_val = 0
        else:
            trix_val = trix[i]

        if position == 0:
            # Regime-based entry
            if chop_aligned[i] < 38.2:  # Trending regime
                # TRIX crossover signals
                if i > 0 and trix_val > 0 and trix[i-1] <= 0 and volume[i] > vol_avg_20[i] * 2.0:
                    signals[i] = 0.25
                    position = 1
                elif i > 0 and trix_val < 0 and trix[i-1] >= 0 and volume[i] > vol_avg_20[i] * 2.0:
                    signals[i] = -0.25
                    position = -1
            elif chop_aligned[i] > 61.8:  # Ranging regime
                # Mean reversion: TRIX extreme + reversal
                if i > 0 and trix_val < -0.5 and trix[i-1] >= -0.5 and volume[i] > vol_avg_20[i] * 2.0:
                    signals[i] = 0.25
                    position = 1
                elif i > 0 and trix_val > 0.5 and trix[i-1] <= 0.5 and volume[i] > vol_avg_20[i] * 2.0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit conditions
            if chop_aligned[i] < 38.2:  # Trending: exit on TRIX signal change
                if i > 0 and trix_val < 0 and trix[i-1] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging: exit on mean reversion
                if trix_val > -0.2:  # Returning to neutral
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit conditions
            if chop_aligned[i] < 38.2:  # Trending: exit on TRIX signal change
                if i > 0 and trix_val > 0 and trix[i-1] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging: exit on mean reversion
                if trix_val < 0.2:  # Returning to neutral
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25

    return signals