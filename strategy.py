#!/usr/bin/env python3
# 6h_1D_KAMA_Trend_Filter_1dTrend_Volume
# Hypothesis: Uses 6h KAMA to capture trend with reduced whipsaw, filtered by 1d trend direction and volume confirmation.
# KAMA adapts to market noise, reducing false signals in sideways markets while capturing strong trends.
# Works in both bull and bear markets by aligning with higher timeframe trend and requiring volume confirmation.

name = "6h_1D_KAMA_Trend_Filter_1dTrend_Volume"
timeframe = "6h"
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

    # Get 6h data for KAMA calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)

    # Calculate Efficiency Ratio (ER) for KAMA
    change = np.abs(np.diff(df_6h['close'], prepend=df_6h['close'][0]))
    volatility = np.abs(np.diff(df_6h['close'], prepend=df_6h['close'][0]))
    er = change / (volatility + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(df_6h['close'])
    kama[0] = df_6h['close'].iloc[0]
    for i in range(1, len(df_6h)):
        kama[i] = kama[i-1] + sc[i] * (df_6h['close'].iloc[i] - kama[i-1])
    kama_6h = kama
    kama_6h_aligned = align_htf_to_ltf(prices, df_6h, kama_6h)

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_6h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filters
        price_above_kama = close[i] > kama_6h_aligned[i]
        price_below_kama = close[i] < kama_6h_aligned[i]
        uptrend_1d = close[i] > ema_34_1d_aligned[i]
        downtrend_1d = close[i] < ema_34_1d_aligned[i]

        if position == 0:
            # LONG: Price above KAMA in 1d uptrend with volume confirmation
            if price_above_kama and uptrend_1d and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA in 1d downtrend with volume confirmation
            elif price_below_kama and downtrend_1d and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or 1d trend turns down
            if price_below_kama or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or 1d trend turns up
            if price_above_kama or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals