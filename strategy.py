#!/usr/bin/env python3
# 4h_MACD_Histogram_Trend_Filter
# Hypothesis: MACD histogram crossing zero with 12h trend filter and volume confirmation captures momentum with low trade frequency.
# Works in bull markets via bullish crosses and in bear via bearish crosses, filtered by higher timeframe trend.
# Target: 15-30 trades/year per symbol.

name = "4h_MACD_Histogram_Trend_Filter"
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
    volume = prices['volume'].values

    # Get 12h data for trend filter (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate MACD (12,26,9)
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        if np.isnan(macd_hist[i]) or np.isnan(signal_line[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: MACD histogram crosses above zero + 12h uptrend + volume spike
            if macd_hist[i] > 0 and macd_hist[i-1] <= 0 and close[i] > ema50_12h_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: MACD histogram crosses below zero + 12h downtrend + volume spike
            elif macd_hist[i] < 0 and macd_hist[i-1] >= 0 and close[i] < ema50_12h_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: MACD histogram crosses below zero or 12h trend turns down
            if macd_hist[i] < 0 and macd_hist[i-1] >= 0 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: MACD histogram crosses above zero or 12h trend turns up
            if macd_hist[i] > 0 and macd_hist[i-1] <= 0 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals