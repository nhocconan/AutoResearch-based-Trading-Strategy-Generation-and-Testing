#!/usr/bin/env python3
# 1d_ThreeLineBreak_Trend_1wTrend_Volume
# Hypothesis: Price reverses after exhaustion shown by Three Line Break (TLB) reversal on 1d,
# confirmed by 1w trend direction and volume spike. Enter on close of reversal bar.
# Works in bull markets (buy after 3-line down reversal in uptrend) and bear markets
# (sell after 3-line up reversal in downtrend). Uses 1w trend filter to avoid counter-trend
# trades and volume spike to confirm institutional participation. Target: 15-30 trades/year.

name = "1d_ThreeLineBreak_Trend_1wTrend_Volume"
timeframe = "1d"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 1w trend: EMA21
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)

    # Three Line Break calculation on daily close
    # Returns +1 for up line, -1 for down line, 0 for no new line
    tl = np.zeros(n, dtype=int)
    if n > 0:
        tl[0] = 1  # start with up line
        line_count = 1
        reversal_level = close[0]
        for i in range(1, n):
            if close[i] > reversal_level:
                tl[i] = 1
                line_count += 1
                reversal_level = close[i]
            elif close[i] < reversal_level:
                tl[i] = -1
                line_count += 1
                reversal_level = close[i]
            else:
                tl[i] = 0

    # Detect TLB reversal: current line opposite to previous line
    # Need at least 3 consecutive same-direction lines for valid reversal signal
    tl_reversal = np.zeros(n, dtype=bool)
    tl_run = np.zeros(n, dtype=int)
    for i in range(1, n):
        if tl[i] == tl[i-1]:
            tl_run[i] = tl_run[i-1] + 1
        else:
            tl_run[i] = 1
        # Reversal when we have at least 3 prior same-direction lines and direction changes
        if i >= 3 and tl_run[i-1] >= 3 and tl[i] != tl[i-1]:
            tl_reversal[i] = True

    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # start after vol MA warmup
        # Skip if any required value is NaN
        if np.isnan(ema21_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TBL down-to-up reversal + 1w uptrend + volume spike
            if tl_reversal[i] and tl[i] == 1 and close[i] > ema21_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TLB up-to-down reversal + 1w downtrend + volume spike
            elif tl_reversal[i] and tl[i] == -1 and close[i] < ema21_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TLB up-to-down reversal or trend failure
            if tl_reversal[i] and tl[i] == -1 or close[i] < ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TLB down-to-up reversal or trend failure
            if tl_reversal[i] and tl[i] == 1 or close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals