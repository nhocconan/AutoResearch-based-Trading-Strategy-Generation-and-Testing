#!/usr/bin/env python3
# 1d_WickReversal_Volume_Trend
# Hypothesis: 1d reversal signals based on long wicks (pin bars) with volume confirmation and 1w trend filter.
# Long wicks indicate rejection of higher/lower prices and potential reversals.
# Volume spike confirms conviction behind the reversal.
# 1w EMA filter ensures we trade with the higher timeframe trend to avoid counter-trend whipsaws.
# Designed for 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull/bear by following 1w trend - in uptrend we look for bullish reversals, in downtrend for bearish reversals.

name = "1d_WickReversal_Volume_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Calculate body and wick sizes
    body = np.abs(close - open_price)
    upper_wick = high - np.maximum(open_price, close)
    lower_wick = np.minimum(open_price, close) - low
    total_range = high - low
    # Avoid division by zero
    total_range = np.where(total_range == 0, 1e-10, total_range)

    # Bullish reversal: long lower wick, small body, close near high
    bullish_wick = (lower_wick > 2 * body) & (close > open_price) & ((high - close) < 0.1 * total_range)
    # Bearish reversal: long upper wick, small body, close near low
    bearish_wick = (upper_wick > 2 * body) & (close < open_price) & ((close - low) < 0.1 * total_range)

    # Volume confirmation: 1.5x average volume
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish reversal in 1w uptrend with volume spike
            if bullish_wick[i] and close[i] > ema20_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish reversal in 1w downtrend with volume spike
            elif bearish_wick[i] and close[i] < ema20_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish reversal signal or price below 1w EMA
            if bearish_wick[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish reversal signal or price above 1w EMA
            if bullish_wick[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals