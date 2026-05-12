#!/usr/bin/env python3
# 12h_Keltner_Channel_Breakout_Volume_Trend
# Hypothesis: Keltner Channel breakout on 12h with volume confirmation and weekly EMA trend filter.
# Price breaking above/below Keltner Channel (EMA-based) indicates momentum expansion.
# Volume spike > 2x average confirms institutional participation.
# Weekly EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Designed for low trade frequency (<30/year) to minimize fee drag in 12h timeframe.
# Works in both bull and bear markets by following the higher timeframe trend.

name = "12h_Keltner_Channel_Breakout_Volume_Trend"
timeframe = "12h"
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

    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate Keltner Channel: EMA(20) +/- ATR(10) * 2
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # EMA20 for middle line
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean()
    
    # True Range for ATR
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr10 = tr.ewm(span=10, adjust=False, min_periods=10).mean()
    
    # Keltner Channel bands
    kc_upper = ema20 + (atr10 * 2)
    kc_lower = ema20 - (atr10 * 2)
    kc_upper_values = kc_upper.values
    kc_lower_values = kc_lower.values
    ema20_values = ema20.values

    # Get weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume spike: 2x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after indicators need sufficient data
        # Skip if any required data is NaN
        if (np.isnan(kc_upper_values[i]) or np.isnan(kc_lower_values[i]) or 
            np.isnan(ema20_values[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Keltner Upper with volume spike and weekly uptrend
            if (close[i] > kc_upper_values[i] and
                volume[i] > volume_spike_threshold[i] and
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Keltner Lower with volume spike and weekly downtrend
            elif (close[i] < kc_lower_values[i] and
                  volume[i] > volume_spike_threshold[i] and
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA20 (middle line)
            if close[i] < ema20_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA20 (middle line)
            if close[i] > ema20_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals