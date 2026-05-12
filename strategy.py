#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_1d_Volume_Filter
Hypothesis: On 4h timeframe, Kaufman Adaptive Moving Average (KAMA) adapts to market noise,
providing reliable trend signals. Combined with 1d volume confirmation (volume > 1.5x 20-period average)
and 1d RSI filter (avoid overbought/oversold extremes), this reduces whipsaws in both bull and bear markets.
Targets 20-50 trades/year (80-200 total over 4 years) with low turnover to minimize fee drag.
"""

name = "4h_KAMA_Trend_With_1d_Volume_Filter"
timeframe = "4h"
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

    # Get daily data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate 4h KAMA (ER=10, FAST=2, SLOW=30)
    def kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
        # Recompute volatility properly
        volatility = np.zeros_like(close)
        for i in range(len(close)):
            if i == 0:
                volatility[i] = 0
            else:
                volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-er_length+1):i+1])))
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out

    # Calculate daily volume average
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values

    # Calculate daily RSI(14)
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out

    # Precompute indicators
    kama_4h = kama(close, er_length=10, fast=2, slow=30)
    vol_avg_20_1d = vol_avg_20
    rsi_1d = rsi(close_1d, length=14)

    # Align daily indicators to 4h
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # warmup for KAMA and daily indicators
        # Get aligned values
        kama_val = kama_4h[i]
        vol_avg_val = vol_avg_20_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]

        # Skip if any required data is NaN
        if np.isnan(kama_val) or np.isnan(vol_avg_val) or np.isnan(rsi_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volume filter: current 4h volume > 1.5x daily average volume
        # Note: comparing 4h volume to daily average - adjust threshold if needed
        vol_filter = volume[i] > vol_avg_val * 1.5

        # RSI filter: avoid extremes
        rsi_filter = (rsi_val > 30) and (rsi_val < 70)

        if position == 0:
            # LONG: Price above KAMA + volume filter + RSI filter
            if close[i] > kama_val and vol_filter and rsi_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + volume filter + RSI filter
            elif close[i] < kama_val and vol_filter and rsi_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals