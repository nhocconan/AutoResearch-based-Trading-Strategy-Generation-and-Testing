#!/usr/bin/env python3
# 4h_EMA_Cross_With_Regime_Filter
# Hypothesis: EMA(21) crossover on 4h timeframe with 12h EMA trend filter and volatility regime filter.
# In trending regimes (ADX > 25), EMA crossovers capture momentum. In ranging regimes (ADX < 20), 
# we avoid trades to prevent whipsaw. Volume confirmation ensures institutional participation.
# Designed for low trade frequency (<30/year) to minimize fee drag in 4h timeframe.

name = "4h_EMA_Cross_With_Regime_Filter"
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

    # Get 12h data for trend filter and ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)

    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values

    # Calculate 4h EMA21 and EMA50 for crossover
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean()
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean()
    ema21_values = ema21.values
    ema50_values = ema50.values

    # Calculate 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean()
    ema34_12h_values = ema34_12h.values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h_values)

    # Calculate 12h ADX(14) for regime filter
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index

    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])

    # Smooth TR and DM
    tr_period = 14
    atr = np.full(len(tr), np.nan)
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)

    # Wilder's smoothing (EMA with alpha = 1/period)
    alpha = 1.0 / tr_period
    for i in range(len(tr)):
        if i == 0:
            atr[i] = tr[i] if not np.isnan(tr[i]) else np.nan
            dm_plus_smooth[i] = dm_plus[i]
            dm_minus_smooth[i] = dm_minus[i]
        else:
            if not np.isnan(tr[i]):
                atr[i] = alpha * tr[i] + (1 - alpha) * (atr[i-1] if not np.isnan(atr[i-1]) else 0)
            dm_plus_smooth[i] = alpha * dm_plus[i] + (1 - alpha) * (dm_plus_smooth[i-1] if not np.isnan(dm_plus_smooth[i-1]) else 0)
            dm_minus_smooth[i] = alpha * dm_minus[i] + (1 - alpha) * (dm_minus_smooth[i-1] if not np.isnan(dm_minus_smooth[i-1]) else 0)

    # DI+ and DI-
    plus_di = 100 * dm_plus_smooth / (atr + 1e-10)
    minus_di = 100 * dm_minus_smooth / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = np.full(len(dx), np.nan)
    for i in range(len(dx)):
        if i < tr_period:
            adx[i] = np.nan
        elif i == tr_period:
            adx[i] = np.nanmean(dx[1:i+1])
        else:
            adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period

    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)

    # Volume spike: 2.0x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 needs 50 bars
        # Skip if any required data is NaN
        if (np.isnan(ema21_values[i]) or np.isnan(ema50_values[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Regime filter: Only trade in trending markets (ADX > 25)
        is_trending = adx_aligned[i] > 25

        if position == 0:
            # LONG: EMA21 crosses above EMA50 with volume spike and uptrend (12h EMA34)
            if (ema21_values[i-1] <= ema50_values[i-1] and 
                ema21_values[i] > ema50_values[i] and
                volume[i] > volume_sma20[i] * 2.0 and
                close[i] > ema34_12h_aligned[i] and
                is_trending):
                signals[i] = 0.25
                position = 1
            # SHORT: EMA21 crosses below EMA50 with volume spike and downtrend (12h EMA34)
            elif (ema21_values[i-1] >= ema50_values[i-1] and 
                  ema21_values[i] < ema50_values[i] and
                  volume[i] > volume_sma20[i] * 2.0 and
                  close[i] < ema34_12h_aligned[i] and
                  is_trending):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: EMA21 crosses below EMA50
            if ema21_values[i] < ema50_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: EMA21 crosses above EMA50
            if ema21_values[i] > ema50_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals