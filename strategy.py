#!/usr/bin/env python3
# 4h_KAMA_Trend_Filter_1dTrend_Volume
# Hypothesis: KAMA adapts to market volatility, providing smooth trend direction. Long when price crosses above KAMA with volume spike and daily uptrend, short when price crosses below KAMA with volume spike and daily downtrend. Designed for 4h timeframe to avoid overtrading. Works in bull markets via trend continuation and in bear markets via trend reversals.

name = "4h_KAMA_Trend_Filter_1dTrend_Volume"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # KAMA (10-period ER, 2 and 30 for fast/slow SC)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10)).values
    volatility = close_s.diff(1).abs().rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(10, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA50
        price_above_daily_ema = close[i] > ema_50_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: Price crosses above KAMA with volume spike and daily uptrend
            if i > 0 and not np.isnan(kama[i-1]) and close[i-1] <= kama[i-1] and close[i] > kama[i] and volume_ok[i] and price_above_daily_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA with volume spike and daily downtrend
            elif i > 0 and not np.isnan(kama[i-1]) and close[i-1] >= kama[i-1] and close[i] < kama[i] and volume_ok[i] and price_below_daily_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or trend turns down
            if i > 0 and not np.isnan(kama[i-1]) and close[i-1] >= kama[i-1] and close[i] < kama[i] or not price_above_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or trend turns up
            if i > 0 and not np.isnan(kama[i-1]) and close[i-1] <= kama[i-1] and close[i] > kama[i] or not price_below_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals