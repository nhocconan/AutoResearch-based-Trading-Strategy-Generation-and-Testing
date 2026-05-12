#!/usr/bin/env python3
# 4h_RSI_Divergence_Confluence_v1
# Hypothesis: Combines RSI divergence with 4h price action and 1d trend filter for high-probability reversals.
# Uses bullish/bearish RSI divergence (price makes new low/high but RSI does not) confirmed by volume spike and 1d EMA trend.
# Designed to work in both bull and bear markets by catching exhaustion moves with strict entry conditions.
# Target: ~20-30 trades/year to minimize fee drag while capturing meaningful reversals.

name = "4h_RSI_Divergence_Confluence_v1"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate RSI (14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values

    # Volume confirmation: 2x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(rsi_values[i-1]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # BULLISH DIVERGENCE: price makes lower low, RSI makes higher low
            bullish_div = (low[i] < low[i-1] and rsi_values[i] > rsi_values[i-1])
            # BEARISH DIVERGENCE: price makes higher high, RSI makes lower high
            bearish_div = (high[i] > high[i-1] and rsi_values[i] < rsi_values[i-1])

            # LONG: Bullish divergence + volume spike + 1d uptrend
            if (bullish_div and
                volume[i] > volume_threshold[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence + volume spike + 1d downtrend
            elif (bearish_div and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence or price breaks below recent swing low
            bearish_div = (high[i] > high[i-1] and rsi_values[i] < rsi_values[i-1])
            if bearish_div or close[i] < low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish divergence or price breaks above recent swing high
            bullish_div = (low[i] < low[i-1] and rsi_values[i] > rsi_values[i-1])
            if bullish_div or close[i] > high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals