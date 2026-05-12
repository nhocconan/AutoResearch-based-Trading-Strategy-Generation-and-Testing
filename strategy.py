#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_Volume_Confirmation
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to capture adaptive trend direction on 4h, combined with volume confirmation and daily trend filter. KAMA adapts to market noise, reducing false signals in chop while capturing trends. Works in both bull/bear markets by requiring volume confirmation and daily trend alignment, avoiding whipsaws. Target: 25-40 trades/year.
"""

name = "4h_KAMA_Trend_With_Volume_Confirmation"
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
    volume = prices['volume'].values

    # Get daily data for trend filter (call once before loop)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)

    # KAMA on 4h (ER period=10, FAST=2, SLOW=30)
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    change = np.concatenate([[np.nan]*10, change])  # align length
    volatility = np.abs(np.diff(close, k=1))  # 1-period volatility
    volatility = np.concatenate([[np.nan], volatility])
    vol_sum10 = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.where(vol_sum10 != 0, change / vol_sum10, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        kama_val = kama[i]
        kama_prev = kama[i-1]
        ema34_val = ema34_daily_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(kama_val) or np.isnan(kama_prev) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price crosses above KAMA + daily uptrend + volume confirmation
            if close[i] > kama_val and close[i-1] <= kama_prev and close[i] > ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: price crosses below KAMA + daily downtrend + volume confirmation
            elif close[i] < kama_val and close[i-1] >= kama_prev and close[i] < ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA or close below daily EMA34
            if close[i] < kama_val or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA or close above daily EMA34
            if close[i] > kama_val or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals