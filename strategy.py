#!/usr/bin/env python3
# 4h_ADX_Keltner_MeanReversion
# Hypothesis: Mean reversion in low-volatility regimes using Keltner Channel (ATR-based) and ADX filter.
# In ranging markets (ADX < 25), price tends to revert to the mean after touching Keltner bands.
# Long when price touches lower band and closes above it; short when touches upper band and closes below.
# Works in both bull/bear markets by focusing on mean reversion rather than trend following.
# Uses 1d trend filter to avoid counter-trend trades in strong trends (ADX > 25 on 1d).

name = "4h_ADX_Keltner_MeanReversion"
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

    # Get 1d data for trend filter and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # ATR for Keltner Channel (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values

    # Keltner Channel (20-period EMA, 2*ATR multiplier)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr

    # ADX calculation (14-period) on 1d for trend filter
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]

    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0

    # Smoothed values
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values

    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)

    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Only trade in low-volatility ranging markets (ADX < 25 on 1d)
        if adx_1d_aligned[i] >= 25:
            # Strong trend - stay flat to avoid whipsaw
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches lower Keltner band and closes above it (mean reversion long)
            if low[i] <= lower_keltner[i] and close[i] > lower_keltner[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches upper Keltner band and closes below it (mean reversion short)
            elif high[i] >= upper_keltner[i] and close[i] < upper_keltner[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches middle (EMA20) or reverses to upper band
            if close[i] >= ema20[i] or high[i] >= upper_keltner[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches middle (EMA20) or reverses to lower band
            if close[i] <= ema20[i] or low[i] <= lower_keltner[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals