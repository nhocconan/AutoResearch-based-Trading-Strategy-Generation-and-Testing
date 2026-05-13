#!/usr/bin/env python3
# 12h_KAMA_RSI_TrendV2
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) identifies trend direction and adapts to volatility.
# Combined with RSI(14) for momentum confirmation and 1d trend filter (EMA50) to avoid counter-trend trades.
# Volume spike (>2x 20-period avg) confirms breakout strength. Designed for 12h timeframe to reduce trade frequency.
# Uses adaptive smoothing to capture trends in both bull and bear markets while minimizing whipsaws.
# Target: 15-35 trades/year per symbol to stay within fee-efficient range.

name = "12h_KAMA_RSI_TrendV2"
timeframe = "12h"
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

    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2) smoothing constant
    slow_sc = 2 / (30 + 1)  # EMA(30) smoothing constant

    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    erosion = np.abs(np.diff(close, n=er_length))
    er = np.divide(erosion, change, out=np.zeros_like(change), where=change!=0)

    # Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))

    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):  # Start after RSI warmup
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA, RSI > 50, uptrend, volume spike
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA, RSI < 50, downtrend, volume spike
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or RSI < 40
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or RSI > 60
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals