#!/usr/bin/env python3
# 4h_1d_KAMA_Trend_RSI_Reversal
# Hypothesis: KAMA determines 4h trend direction (bullish/bearish), RSI identifies overbought/oversold conditions for mean reversion entries in the direction of the trend. Works in both bull and bear markets by following the trend filter.
# Uses 1d timeframe for trend confirmation and RSI calculation to reduce noise and avoid overtrading. Designed for 20-50 trades/year.

name = "4h_1d_KAMA_Trend_RSI_Reversal"
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
    volume = prices['volume'].values

    # Get daily data for trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily KAMA for trend filter
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility as sum of absolute changes over 10 periods
    volatility = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 10:
            volatility[i] = np.sum(np.abs(np.diff(close_1d[:i+1]))) if i > 0 else 0
        else:
            volatility[i] = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama

    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = rsi

    # Align KAMA and RSI to 4h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine trend based on price vs KAMA
        bullish_trend = close[i] > kama_1d_aligned[i]
        bearish_trend = close[i] < kama_1d_aligned[i]

        if position == 0:
            # LONG: Oversold RSI (<30) in bullish trend with volume confirmation
            if rsi_1d_aligned[i] < 30 and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Overbought RSI (>70) in bearish trend with volume confirmation
            elif rsi_1d_aligned[i] > 70 and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought (>70) or trend turns bearish
            if rsi_1d_aligned[i] > 70 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold (<30) or trend turns bullish
            if rsi_1d_aligned[i] < 30 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals