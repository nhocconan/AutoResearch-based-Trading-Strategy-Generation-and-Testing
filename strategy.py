#!/usr/bin/env python3
# 1d_KAMA_Trend_Follow_With_Chop_Filter
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) on daily timeframe provides trend direction.
# Chopiness Index (14) filters for trending markets (CHOP < 38.2) to avoid whipsaws in ranging conditions.
# Entry when price crosses KAMA with volume confirmation (>1.5x 20-day average volume).
# Designed for low trade frequency (<25/year) to minimize fee drag in 1d timeframe.
# Works in both bull and bear markets by following trend only when market is trending (low chop).

name = "1d_KAMA_Trend_Follow_With_Chop_Filter"
timeframe = "1d"
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

    # Get weekly data for Chopiness Index filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate KAMA on daily close
    close_series = pd.Series(close)
    # Efficiency Ratio (ER) over 10 periods
    change = abs(close - np.roll(close, 10))
    change[0:10] = np.nan  # First 10 values invalid
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # Will be calculated properly below
    # Recalculate volatility as sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # Fast=2, Slow=30
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Calculate Chopiness Index on weekly data
    def true_range(high, low, close_prev):
        return np.maximum(np.maximum(high - low, np.abs(high - close_prev)), np.abs(low - close_prev))

    tr1w = np.zeros_like(high_1w)
    tr1w[0] = high_1w[0] - low_1w[0]  # First TR
    for i in range(1, len(high_1w)):
        tr1w[i] = true_range(high_1w[i], low_1w[i], close_1w[i-1])

    # Sum of TR over 14 periods
    atr1w = np.zeros_like(high_1w)
    for i in range(13, len(high_1w)):
        atr1w[i] = np.sum(tr1w[i-13:i+1])

    # Calculate Chopiness Index: 100 * log10(ATR14 / (sum of ranges)) / log10(14)
    ranges1w = high_1w - low_1w
    sum_ranges1w = np.zeros_like(high_1w)
    for i in range(13, len(high_1w)):
        sum_ranges1w[i] = np.sum(ranges1w[i-13:i+1])

    chop = np.zeros_like(high_1w)
    for i in range(13, len(high_1w)):
        if sum_ranges1w[i] > 0:
            chop[i] = 100 * np.log10(atr1w[i] / sum_ranges1w[i]) / np.log10(14)
        else:
            chop[i] = 50  # Default to neutral when undefined

    # Align Chop to daily timeframe (with 2-bar delay for weekly confirmation)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=2)

    # KAMA aligned to daily (no additional delay needed for trend)
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)

    # Volume confirmation: 1.5x 20-day average volume
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_sma20[i]) or np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above KAMA, low chop (trending market), volume confirmation
            if (close[i-1] <= kama_aligned[i-1] and close[i] > kama_aligned[i] and
                chop_aligned[i] < 38.2 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA, low chop (trending market), volume confirmation
            elif (close[i-1] >= kama_aligned[i-1] and close[i] < kama_aligned[i] and
                  chop_aligned[i] < 38.2 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA OR chop becomes too high (ranging market)
            if (close[i] < kama_aligned[i]) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA OR chop becomes too high (ranging market)
            if (close[i] > kama_aligned[i]) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals