#!/usr/bin/env python3
"""
4h_Momentum_Reversal_Confluence
Hypothesis: Combines momentum (RSI divergence) with mean reversion (BB reversal) on 4h timeframe.
Uses 1d trend filter (EMA50) to align with higher timeframe direction, reducing counter-trend trades.
Volume confirmation ensures momentum validity. Designed to capture reversals in both bull and bear markets
by requiring alignment between momentum exhaustion and price rejection at Bollinger Bands.
Target: 20-40 trades/year to minimize fee drag while maintaining edge.
"""

name = "4h_Momentum_Reversal_Confluence"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # RSI for momentum (14-period)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    upper = upper.values
    lower = lower.values

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean()
    volume_threshold = volume_sma20 * 1.5
    volume_threshold = volume_threshold.values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        ema50_aligned = ema50_1d_aligned[i]
        vol_threshold_val = volume_threshold[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50_aligned) or np.isnan(vol_threshold_val) or
            np.isnan(rsi[i]) or np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI oversold (<30) + price at/below lower BB + volume spike + 1d uptrend
            if (rsi[i] < 30 and
                close[i] <= lower[i] and
                volume[i] > vol_threshold_val and
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) + price at/above upper BB + volume spike + 1d downtrend
            elif (rsi[i] > 70 and
                  close[i] >= upper[i] and
                  volume[i] > vol_threshold_val and
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses above 50 (momentum shift) or price reaches middle BB
            middle = (upper[i] + lower[i]) / 2
            if rsi[i] > 50 or close[i] >= middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses below 50 or price reaches middle BB
            middle = (upper[i] + lower[i]) / 2
            if rsi[i] < 50 or close[i] <= middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals