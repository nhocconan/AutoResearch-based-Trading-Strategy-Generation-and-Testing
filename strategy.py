#!/usr/bin/env python3
# 4h_Trix_VolumeRegime
# Hypothesis: TRIX (1-period smoothed triple EMA) + volume spike + chop regime (ADX < 25) identifies momentum bursts in low-volatility environments.
# Long: TRIX crosses above zero, volume spike, ADX < 25. Short: TRIX crosses below zero, volume spike, ADX < 25.
# Exit: TRIX crosses back through zero or ADX rises above 25 (trending regime).
# Works in both bull and bear markets by capturing momentum bursts during consolidation.
# Target: 20-50 trades/year per symbol.

name = "4h_Trix_VolumeRegime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values

    # TRIX: 1-period smoothed triple EMA (15-period EMA applied 3 times)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.values

    # ADX for chop regime (ADX < 25 = ranging/choppy)
    # Calculate +DI, -DI, DX
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar has no previous close

    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0

    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx[:13] = np.nan  # not enough data for first 14 periods

    # Chop regime: ADX < 25
    chop_regime = adx < 25

    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(15, n):  # start after TRIX warmup
        # Skip if any required value is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(trix[i-1]) or 
            np.isnan(chop_regime[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero, volume spike, chop regime
            if trix[i] > 0 and trix[i-1] <= 0 and volume_spike[i] and chop_regime[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero, volume spike, chop regime
            elif trix[i] < 0 and trix[i-1] >= 0 and volume_spike[i] and chop_regime[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or regime changes (ADX >= 25)
            if trix[i] < 0 or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or regime changes (ADX >= 25)
            if trix[i] > 0 or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals