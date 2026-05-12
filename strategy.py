#!/usr/bin/env python3
# 1d_1W_KAMA_Trend_Filter_1wTrend_Volume
# Hypothesis: KAMA trend on daily timeframe with weekly trend filter and volume confirmation.
# In bull markets: go long when KAMA turns up and price > KAMA in uptrend.
# In bear markets: go short when KAMA turns down and price < KAMA in downtrend.
# Volume confirms trend strength. Weekly trend filter avoids counter-trend trades.
# Designed for 1d to limit trade frequency (target: 7-25 trades/year).

name = "1d_1W_KAMA_Trend_Filter_1wTrend_Volume"
timeframe = "1d"
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

    # Get daily data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # KAMA calculation (ER=10, fast=2, slow=30)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # placeholder, will fix below
    # Recalculate volatility properly: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 10:
            volatility[i] = np.sum(np.abs(np.diff(close_1d[:i+1], prepend=close_1d[0])))
        else:
            volatility[i] = np.sum(np.abs(np.diff(close_1d[i-9:i+1], prepend=close_1d[i-9])))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume confirmation: current volume > 1.5x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Daily KAMA trend
        price_above_kama = close[i] > kama_1d[i]
        price_below_kama = close[i] < kama_1d[i]

        # Weekly trend filter
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]

        if position == 0:
            # LONG: Price above KAMA in weekly uptrend with volume confirmation
            if price_above_kama and weekly_uptrend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA in weekly downtrend with volume confirmation
            elif price_below_kama and weekly_downtrend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or weekly trend turns down
            if price_below_kama or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or weekly trend turns up
            if price_above_kama or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals