#!/usr/bin/env python3
# 12h_ParabolicSAR_1dTrend_Filter
# Hypothesis: Parabolic SAR on 12h with 1d EMA trend filter for trend-following entries.
# Works in bull/bear by following 1d trend; SAR provides dynamic stop/reversal.
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "12h_ParabolicSAR_1dTrend_Filter"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')

    # Parabolic SAR on 12h
    # Initialize
    psar = np.zeros(n)
    psar[0] = low[0]
    trend = 1  # 1 for up, -1 for down
    af = 0.02  # acceleration factor
    max_af = 0.2
    ep = high[0] if trend == 1 else low[0]  # extreme point

    for i in range(1, n):
        psar[i] = psar[i-1] + af * (ep - psar[i-1])
        # Reverse trend if price crosses SAR
        if trend == 1 and low[i] < psar[i]:
            trend = -1
            psar[i] = ep
            af = 0.02
            ep = low[i]
        elif trend == -1 and high[i] > psar[i]:
            trend = 1
            psar[i] = ep
            af = 0.02
            ep = high[i]
        else:
            # Update extreme point and acceleration factor
            if trend == 1:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
            psar[i] = psar[i-1] + af * (ep - psar[i-1])

    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        if (np.isnan(psar[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above SAR and above 1d EMA50
            if close[i] > psar[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below SAR and below 1d EMA50
            elif close[i] < psar[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below SAR or trend turns down
            if close[i] < psar[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above SAR or trend turns up
            if close[i] > psar[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals