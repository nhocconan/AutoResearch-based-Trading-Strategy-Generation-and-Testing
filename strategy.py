#!/usr/bin/env python3
# 4h_1d_TRIX_Zero_Cross_With_Volume_Spike_And_Chop_Filter
# Hypothesis: TRIX (Triple Exponential Average) crossing zero on 1d timeframe provides momentum direction,
# confirmed by volume spike (>1.5x 20-period average) on 4h and filtered by Choppiness Index (>61.8 = ranging) to avoid whipsaws.
# Works in bull/bear by following 1d TRIX trend; volume confirms breakout strength; chop filter reduces false signals.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_1d_TRIX_Zero_Cross_With_Volume_Spike_And_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for TRIX calculation (15-period EMA applied 3x)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    # Calculate TRIX: EMA(EMA(EMA(close, 15), 15), 15) then % change
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix_raw = ema3.pct_change() * 100  # Convert to percentage
    trix = trix_raw.fillna(0).values  # Handle NaN from pct_change
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix)

    # Get 4h data for Choppiness Index
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate Choppiness Index (14-period)
    atr = []
    for i in range(len(df_4h)):
        if i == 0:
            tr = high_4h[i] - low_4h[i]
        else:
            tr = max(high_4h[i] - low_4h[i],
                     abs(high_4h[i] - close_4h[i-1]),
                     abs(low_4h[i] - close_4h[i-1]))
        atr.append(tr)
    
    atr = np.array(atr)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    chop = np.where(range_hl != 0, 100 * np.log10(atr_sum / range_hl) / np.log10(14), 50)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop)

    # Calculate 4h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(trix_1d_aligned[i]) or np.isnan(chop_4h_aligned[i]) or
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero, in low chop (trending), with volume spike
            if (trix_1d_aligned[i] > 0 and trix_1d_aligned[i-1] <= 0 and
                chop_4h_aligned[i] < 61.8 and volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero, in low chop (trending), with volume spike
            elif (trix_1d_aligned[i] < 0 and trix_1d_aligned[i-1] >= 0 and
                  chop_4h_aligned[i] < 61.8 and volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR high chop (ranging market)
            if (trix_1d_aligned[i] < 0 and trix_1d_aligned[i-1] >= 0) or chop_4h_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR high chop (ranging market)
            if (trix_1d_aligned[i] > 0 and trix_1d_aligned[i-1] <= 0) or chop_4h_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals