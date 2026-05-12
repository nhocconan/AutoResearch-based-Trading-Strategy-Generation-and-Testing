#!/usr/bin/env python3
# 4h_Keltner_Channel_Breakout_Volume_Filter
# Hypothesis: Keltner Channel breakout with volume confirmation and 1d EMA trend filter on 4h timeframe.
# Uses ATR-based channels to capture volatility breakouts, filters by trend direction from higher timeframe EMA,
# and requires volume spike for confirmation. Designed for low trade frequency (<50/year) to minimize fee drag.
# Works in bull markets by following uptrend breakouts, in bear markets by following downtrend breakouts.
# Exit on opposite channel touch to avoid whipsaw.

name = "4h_Keltner_Channel_Breakout_Volume_Filter"
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

    # Get 4h data for Keltner Channel calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate ATR(10) for Keltner Channel width
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Calculate EMA(20) for Keltner Channel middle line
    ema20 = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Calculate Keltner Channel bands
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr

    # Align Keltner Channel bands to lower timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_4h, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_4h, kc_lower)

    # Get 1d data for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate volume spike threshold (1.5x 20-period SMA)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kc_upper_aligned[i]) or np.isnan(kc_lower_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Keltner Channel in uptrend with volume spike
            if (close[i] > kc_upper_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner Channel in downtrend with volume spike
            elif (close[i] < kc_lower_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches or crosses below lower Keltner Channel
            if close[i] < kc_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches or crosses above upper Keltner Channel
            if close[i] > kc_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals