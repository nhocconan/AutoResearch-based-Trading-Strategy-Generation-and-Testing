#!/usr/bin/env python3
# 6h_Stochastic_RSI_WeeklyTrend
# Hypothesis: Use Stochastic RSI on 6h for mean-reversion entries in oversold/overbought conditions,
# filtered by weekly trend (EMA200) and volume confirmation. Enter long when StochRSI < 0.2,
# price above weekly EMA200, and volume spike; short when StochRSI > 0.8, price below weekly EMA200,
# and volume spike. Exit on StochRSI crossing back to neutral (0.5). Targets 20-40 trades/year
# to minimize fee drag and profit from mean reversion in ranging markets while respecting weekly trend.

name = "6h_Stochastic_RSI_WeeklyTrend"
timeframe = "6h"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate Stochastic RSI (14,14,3,3) on 6h
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Stochastic RSI: (RSI - min(RSI)) / (max(RSI) - min(RSI)) over 14 periods
    rsi_series = pd.Series(rsi)
    min_rsi = rsi_series.rolling(window=14, min_periods=14).min().values
    max_rsi = rsi_series.rolling(window=14, min_periods=14).max().values
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-10)

    # 200 EMA on weekly for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(stoch_rsi[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: StochRSI oversold (<0.2) + price > weekly EMA200 + volume spike
            if (stoch_rsi[i] < 0.2 and 
                close[i] > ema200_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: StochRSI overbought (>0.8) + price < weekly EMA200 + volume spike
            elif (stoch_rsi[i] > 0.8 and 
                  close[i] < ema200_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: StochRSI crosses back above 0.5 (mean reversion complete)
            if stoch_rsi[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: StochRSI crosses back below 0.5
            if stoch_rsi[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals