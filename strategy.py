#!/usr/bin/env python3

# 4h_1D_RSI_Trend_Momentum
# Hypothesis: Enter long when price closes above 40-period EMA with RSI(14) > 55 on 4h and bullish 1d trend (price > 50-period EMA).
# Enter short when price closes below 40-period EMA with RSI(14) < 45 on 4h and bearish 1d trend (price < 50-period EMA).
# Uses trend alignment to filter counter-trend trades and RSI for momentum confirmation.
# Designed for 4h timeframe with ~20-40 trades/year to avoid overtrading.

name = "4h_1D_RSI_Trend_Momentum"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 1d 50-period EMA for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate 40-period EMA on 4h
    ema_40 = pd.Series(close).ewm(span=40, adjust=False, min_periods=40).mean().values

    # Calculate 14-period RSI on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_40[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 50-period EMA on 1d
        bullish_trend = close[i] > ema_50_1d_aligned[i]
        bearish_trend = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: Price above 40 EMA, RSI > 55, bullish 1d trend
            if close[i] > ema_40[i] and rsi[i] > 55 and bullish_trend:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below 40 EMA, RSI < 45, bearish 1d trend
            elif close[i] < ema_40[i] and rsi[i] < 45 and bearish_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below 40 EMA or RSI < 50 or bearish 1d trend
            if close[i] < ema_40[i] or rsi[i] < 50 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above 40 EMA or RSI > 50 or bullish 1d trend
            if close[i] > ema_40[i] or rsi[i] > 50 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals