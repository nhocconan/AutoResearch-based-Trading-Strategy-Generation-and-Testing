#!/usr/bin/env python3

# 1h_4h_1D_Combined_Strategy
# Hypothesis: Combine 4h trend direction (EMA50) with 1h momentum (RSI pullback) and volume confirmation.
# Uses 4h EMA50 for trend filter (works in bull/bear by aligning with higher timeframe trend),
# 1h RSI for oversold/overbought entries during pullbacks, and volume spike for confirmation.
# Targets 15-30 trades/year by requiring multiple confluence factors.

name = "1h_4h_1D_Combined_Strategy"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)

    # Get 1d data for regime filter (optional trend strength)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 1d EMA200 for long-term trend
    close_1d = df_1d['close'].values
    ema_1d_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)

    # 1h RSI (14-period) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: current volume > 2.0x average of last 24 periods
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_200_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filters: 4h EMA50 and 1d EMA200
        bullish_trend_4h = close[i] > ema_4h_aligned[i]
        bearish_trend_4h = close[i] < ema_4h_aligned[i]
        bullish_trend_1d = close[i] > ema_1d_200_aligned[i]
        bearish_trend_1d = close[i] < ema_1d_200_aligned[i]

        if position == 0:
            # LONG: Pullback in uptrend - RSI oversold with volume
            if (bullish_trend_4h and bullish_trend_1d and
                rsi[i] < 30 and volume_ok[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Pullback in downtrend - RSI overbought with volume
            elif (bearish_trend_4h and bearish_trend_1d and
                  rsi[i] > 70 and volume_ok[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought or trend breaks
            if rsi[i] > 70 or not (bullish_trend_4h and bullish_trend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI oversold or trend breaks
            if rsi[i] < 30 or not (bearish_trend_4h and bearish_trend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals