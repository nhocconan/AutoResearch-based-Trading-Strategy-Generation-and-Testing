#!/usr/bin/env python3
# 1h_HeikinAshi_Trend_Breakout_4h1d
# Hypothesis: Use Heikin-Ashi smoothed candles to filter noise on 1h, with 4h trend and 1d momentum confirmation.
# Long when: HA close > HA open (bullish candle), price > 4h EMA50, and 1d RSI > 50.
# Short when: HA close < HA open (bearish candle), price < 4h EMA50, and 1d RSI < 50.
# Exit when HA candle reverses color.
# Designed for 1-3 trades per week, targeting 15-30/year to avoid fee drag.
# Works in bull (trend continuation) and bear (mean reversion via RSI filter).

name = "1h_HeikinAshi_Trend_Breakout_4h1d"
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

    # Calculate Heikin-Ashi
    ha_close = (high + low + close + close) / 4.0
    ha_open = np.zeros(n)
    ha_open[0] = (high[0] + low[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
    ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(low, np.minimum(ha_open, ha_close))

    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # 1d RSI for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        # Skip if any required value is NaN
        if (np.isnan(ha_open[i]) or np.isnan(ha_close[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        ha_bullish = ha_close[i] > ha_open[i]
        ha_bearish = ha_close[i] < ha_open[i]

        if position == 0:
            # LONG: Bullish HA + price > 4h EMA50 + 1d RSI > 50
            if ha_bullish and close[i] > ema50_4h_aligned[i] and rsi_1d_aligned[i] > 50:
                signals[i] = 0.20
                position = 1
            # SHORT: Bearish HA + price < 4h EMA50 + 1d RSI < 50
            elif ha_bearish and close[i] < ema50_4h_aligned[i] and rsi_1d_aligned[i] < 50:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: HA turns bearish
            if ha_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: HA turns bullish
            if ha_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals