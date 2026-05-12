#!/usr/bin/env python3
"""
6h_Poisson_Trend_Weakness
Hypothesis: In strong trends, price moves in bursts with low entropy; weakening trends show increased randomness.
We count positive/negative returns over 6 periods (Poisson-like rate). Strong uptrend when positive returns > 2x negative.
Strong downtrend when negative returns > 2x positive. Enter on weakness: when the dominant trend rate drops below threshold.
Uses 1w trend filter to avoid counter-trend trades. Works in bull/bear by following the dominant weekly trend.
"""

name = "6h_Poisson_Trend_Weakness"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate returns over 6 periods (approx 1.5 days at 6h)
    returns = np.diff(np.log(close), prepend=0)
    pos_returns = np.where(returns > 0, returns, 0)
    neg_returns = np.where(returns < 0, -returns, 0)

    # Sum of positive/negative returns over 6-bar window (rate parameter)
    sum_pos = pd.Series(pos_returns).rolling(window=6, min_periods=6).sum().values
    sum_neg = pd.Series(neg_returns).rolling(window=6, min_periods=6).sum().values

    # Trend strength ratio: avoid division by zero
    ratio = np.where(sum_neg > 0, sum_pos / sum_neg, np.inf)
    ratio_inv = np.where(sum_pos > 0, sum_neg / sum_pos, np.inf)

    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume filter: avoid low-liquidity periods
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_24[i]) or np.isnan(ratio[i]) or np.isnan(ratio_inv[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Strong uptrend weakening (bulls losing momentum) in 1w uptrend
            if (ratio[i] > 2.0 and ratio[i] < ratio[i-1] and  # weakening uptrend
                close[i] > ema50_1w_aligned[i] and              # 1w uptrend filter
                volume[i] > vol_avg_24[i] * 1.1):               # volume confirmation
                signals[i] = 0.25
                position = 1
            # SHORT: Strong downtrend weakening (bears losing momentum) in 1w downtrend
            elif (ratio_inv[i] > 2.0 and ratio_inv[i] < ratio_inv[i-1] and  # weakening downtrend
                  close[i] < ema50_1w_aligned[i] and                        # 1w downtrend filter
                  volume[i] > vol_avg_24[i] * 1.1):                         # volume confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakness ends or 1w trend turns down
            if (ratio[i] <= ratio[i-1] or  # weakness stopped
                close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakness ends or 1w trend turns up
            if (ratio_inv[i] <= ratio_inv[i-1] or  # weakness stopped
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals