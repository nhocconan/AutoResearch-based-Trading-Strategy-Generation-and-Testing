#!/usr/bin/env python3
# 4h_4x4_Squeeze_Momentum
# Hypothesis: Squeeze momentum with Bollinger Bands and Keltner Channel identifies low volatility breakouts.
# Combines with 1d trend filter (EMA50) and volume confirmation for high-probability entries.
# Designed for low-frequency, high-quality setups in both bull and bear markets.

name = "4h_4x4_Squeeze_Momentum"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Bollinger Bands (20, 2)
    bb_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std

    # Keltner Channel (20, 1.5)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = bb_ma + 1.5 * atr
    keltner_lower = bb_ma - 1.5 * atr

    # Squeeze condition: BB inside KC
    squeeze = (bb_lower > keltner_lower) & (bb_upper < keltner_upper)

    # Momentum: close - BB middle
    momentum = close - bb_ma

    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_conf = volume > 1.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(momentum[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Squeeze release + bullish momentum + uptrend + volume
            if (not squeeze[i-1] and squeeze[i]) and momentum[i] > 0 and close[i] > ema50_1d_aligned[i] and volume_conf[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Squeeze release + bearish momentum + downtrend + volume
            elif (not squeeze[i-1] and squeeze[i]) and momentum[i] < 0 and close[i] < ema50_1d_aligned[i] and volume_conf[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Squeeze fires or momentum turns bearish
            if squeeze[i] or momentum[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Squeeze fires or momentum turns bullish
            if squeeze[i] or momentum[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals