#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_1dTrend_With_Volume
# Hypothesis: Use KAMA to determine trend direction on 4h, enter long when price pulls back to KAMA during uptrend with RSI < 40 and volume > 1.5x average, short when price pulls back to KAMA during downtrend with RSI > 60 and volume > 1.5x average.
# Uses 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades.
# Designed for low frequency: only trades on pullbacks in strong trends with volume confirmation.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).

name = "4h_KAMA_Direction_RSI_1dTrend_With_Volume"
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

    # 4h KAMA (trend direction)
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))  # 10-period change
    abs_change = np.sum(np.abs(np.diff(close, k=1)), axis=1) if len(close) > 1 else 0  # placeholder for actual ER calc
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            direction = np.abs(close[i] - close[i-10])
            volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # 4h RSI (14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length
    rsi = np.concatenate([np.full(14, np.nan), rsi])

    # 4h volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # start after warmup
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price near KAMA (pullback) in uptrend with oversold RSI and volume spike
            if (close[i] <= kama[i] * 1.005 and  # within 0.5% above KAMA
                close[i] > kama[i] and          # above KAMA
                close[i] > ema50_1d_aligned[i] and  # above daily EMA50 (uptrend)
                rsi[i] < 40 and               # oversold
                volume[i] > 1.5 * vol_ma_20[i]):  # volume spike
                signals[i] = 0.25
                position = 1
            # SHORT: Price near KAMA (pullback) in downtrend with overbought RSI and volume spike
            elif (close[i] >= kama[i] * 0.995 and  # within 0.5% below KAMA
                  close[i] < kama[i] and           # below KAMA
                  close[i] < ema50_1d_aligned[i] and  # below daily EMA50 (downtrend)
                  rsi[i] > 60 and                # overbought
                  volume[i] > 1.5 * vol_ma_20[i]):  # volume spike
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or RSI overbought
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or RSI oversold
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals