#!/usr/bin/env python3
# 1h_Bollinger_Bands_Mean_Reversion_4hTrend_Filter
# Hypothesis: In 1h timeframe, price often mean-reverts at Bollinger Band extremes (±2σ) during range-bound markets. Use 4h EMA50 as trend filter: only take long when price > 4h EMA50 (uptrend bias) and short when price < 4h EMA50 (downtrend bias). This avoids fighting the trend. Entry: price touches lower/upper BB + volume > 1.5x 20-period average. Exit: price crosses back to 20-period SMA. Targets 20-40 trades/year to minimize fee drag while capturing mean reversion in both bull and bear markets.

name = "1h_Bollinger_Bands_Mean_Reversion_4hTrend_Filter"
timeframe = "1h"
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

    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values

    # Calculate Bollinger Bands (20, 2) on 1h close
    close_s = pd.Series(close)
    sma20 = close_s.rolling(window=20, min_periods=20).mean()
    std20 = close_s.rolling(window=20, min_periods=20).std()
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20

    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches lower BB + price > 4h EMA50 (uptrend bias) + volume spike
            if (close[i] <= lower_band[i] and 
                close[i] > ema50_4h_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price touches upper BB + price < 4h EMA50 (downtrend bias) + volume spike
            elif (close[i] >= upper_band[i] and 
                  close[i] < ema50_4h_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back to 20-period SMA (mean reversion complete)
            if close[i] >= sma20.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses back to 20-period SMA (mean reversion complete)
            if close[i] <= sma20.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals