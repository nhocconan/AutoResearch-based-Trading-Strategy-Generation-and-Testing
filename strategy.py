#!/usr/bin/env python3

# 4h_1D_TRIX_VolumeSpike_Regime
# Hypothesis: TRIX momentum on 4h with 1d trend filter, volume spike, and Choppiness regime filter.
# TRIX filters noise and identifies momentum shifts. Volume spike confirms conviction.
# Choppiness regime filter avoids whipsaws in ranging markets. Works in bull/bear by requiring
# alignment with 1d trend and volatility regime. Targets 20-40 trades/year.

name = "4h_1D_TRIX_VolumeSpike_Regime"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate TRIX on 4h (15-period EMA of 15-period EMA of 15-period EMA)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # First value has no previous

    # Volume confirmation: current volume > 2.0x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    # Choppiness regime filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop[np.isnan(chop) | np.isinf(chop)] = 50  # Default to neutral
    chop[highest_high - lowest_low == 0] = 50   # Avoid division by zero
    chop_trending = chop < 38.2  # Trending regime
    chop_ranging = chop > 61.8   # Ranging regime

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(15, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(trix[i]) or
            np.isnan(volume_ok[i]) or np.isnan(chop_trending[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 1d
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: TRIX turning up in trending regime with bullish trend and volume spike
            if trix[i] > trix[i-1] and trix[i] > 0 and chop_trending[i] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX turning down in trending regime with bearish trend and volume spike
            elif trix[i] < trix[i-1] and trix[i] < 0 and chop_trending[i] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns down OR trend turns bearish OR enters ranging market
            if trix[i] < trix[i-1] or not bullish_trend or chop_ranging[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns up OR trend turns bullish OR enters ranging market
            if trix[i] > trix[i-1] or not bearish_trend or chop_ranging[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals