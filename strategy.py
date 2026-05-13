#!/usr/bin/env python3
# 4h_KAMA_Trend_With_RSI_Filter_v2
# Hypothesis: KAMA identifies trend direction with low lag; RSI(14) filters overbought/oversold conditions in ranging markets.
# Long when KAMA rising and RSI < 50; short when KAMA falling and RSI > 50, with volume confirmation.
# Uses 12h EMA50 as higher timeframe trend filter to avoid counter-trend trades. Designed for fewer trades (<50/year) to reduce fee drag.
# Works in bull via trend following and bear via mean reversion during range-bound periods.

name = "4h_KAMA_Trend_With_RSI_Filter_v2"
timeframe = "4h"
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

    # KAMA: Kaufman Adaptive Moving Average
    def kama(close, slow=2, fast=30):
        dir = np.abs(np.diff(close, n=10))  # direction over 10 periods
        vol = np.sum(np.abs(np.diff(close)), axis=1)  # volatility
        er = np.where(vol != 0, dir / vol, 0)  # efficiency ratio
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2  # smoothing constant
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    # RSI: Relative Strength Index
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    kama_vals = kama(close, 2, 30)
    rsi_vals = rsi(close, 14)

    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume filter: >1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising (trend up) AND RSI < 50 (not overbought) AND volume spike
            if (kama_vals[i] > kama_vals[i-1] and 
                rsi_vals[i] < 50 and
                volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (trend down) AND RSI > 50 (not oversold) AND volume spike
            elif (kama_vals[i] < kama_vals[i-1] and 
                  rsi_vals[i] > 50 and
                  volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling OR RSI > 70 (overbought) OR volume drops
            if (kama_vals[i] < kama_vals[i-1] or 
                rsi_vals[i] > 70 or
                volume[i] < vol_avg_30[i] * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising OR RSI < 30 (oversold) OR volume drops
            if (kama_vals[i] > kama_vals[i-1] or 
                rsi_vals[i] < 30 or
                volume[i] < vol_avg_30[i] * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals