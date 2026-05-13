#!/usr/bin/env python3
# 6h_ConnorsRSI_MeanReversion_1dTrend_Volume
# Hypothesis: Use Connors RSI (RSI(3) + RSI of streak + PercentRank(100))/3 for mean reversion on 6b timeframe.
# Long when CRSI < 15 and price > 1d EMA50 (uptrend), short when CRSI > 85 and price < 1d EMA50 (downtrend).
# Exit when CRSI crosses above 70 (long) or below 30 (short) or price crosses 1d EMA50.
# Volume confirmation: volume > 1.5x 20-period average to avoid low-liquidity false signals.
# Designed for 6h timeframe to achieve 15-25 trades/year, targeting 60-100 total over 4 years.
# Works in bull via uptrend longs, in bear via downtrend shorts, with mean reversion edges in ranging markets.

name = "6h_ConnorsRSI_MeanReversion_1dTrend_Volume"
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
    volume = prices['volume'].values

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate Connors RSI on close prices
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    loss_ma = pd.Series(loss).ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    rs = gain_ma / (loss_ma + 1e-10)
    rsi_3 = 100 - (100 / (1 + rs))

    # Streak RSI: RSI of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    streak_abs = np.abs(streak)
    # RSI of streak (capped at 4 for calculation)
    streak_limited = np.minimum(streak_abs, 4)
    delta_streak = np.diff(streak_limited, prepend=streak_limited[0])
    gain_streak = np.where(delta_streak > 0, delta_streak, 0)
    loss_streak = np.where(delta_streak < 0, -delta_streak, 0)
    gain_ma_streak = pd.Series(gain_streak).ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    loss_ma_streak = pd.Series(loss_streak).ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    rs_streak = gain_ma_streak / (loss_ma_streak + 1e-10)
    rsi_streak = 100 - (100 / (1 + rs_streak))

    # Percent Rank(100): where current close ranks in last 100 periods
    def percentile_of_score(arr, score):
        if len(arr) == 0:
            return 50.0
        return (np.sum(arr < score) + 0.5 * np.sum(arr == score)) / len(arr) * 100

    pr_100 = np.full(n, np.nan)
    for i in range(99, n):
        window = close[i-99:i+1]
        pr_100[i] = percentile_of_score(window[:-1], close[i])

    # Connors RSI = (RSI(3) + RSI_streak + PercentRank(100)) / 3
    crsi = (rsi_3 + rsi_streak + pr_100) / 3.0

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Start after PR(100) window
        # Skip if any required value is NaN
        if (np.isnan(crsi[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: CRSI < 15 (oversold) + price > 1d EMA50 (uptrend) + volume spike
            if (crsi[i] < 15 and 
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: CRSI > 85 (overbought) + price < 1d EMA50 (downtrend) + volume spike
            elif (crsi[i] > 85 and 
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CRSI > 70 (overbought threshold) OR price crosses below 1d EMA50
            if crsi[i] > 70 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CRSI < 30 (oversold threshold) OR price crosses above 1d EMA50
            if crsi[i] < 30 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals