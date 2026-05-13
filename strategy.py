#!/usr/bin/env python3
# 1d_Weekly_KAMA_Trend_Filter
# Hypothesis: KAMA adapts to market efficiency, reducing lag in trends and noise in ranges.
# Combined with weekly trend filter (price above/below weekly KAMA) and volume confirmation,
# this strategy captures sustained moves while avoiding whipsaw. Weekly trend ensures alignment
# with higher-timeframe momentum. Daily KAMA crossovers provide timely entries/exits.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "1d_Weekly_KAMA_Trend_Filter"
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
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate weekly KAMA for trend filter
    def calculate_kama(close, length=30, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum()
        volatility = np.where(volatility == 0, 1, volatility)
        er = change / volatility
        er = np.where(np.isnan(er), 0, er)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    wkama = calculate_kama(close_1w, length=30, fast=2, slow=30)
    wkama_aligned = align_htf_to_ltf(prices, df_1w, wkama)

    # Calculate daily KAMA for entry/exit
    dkama = calculate_kama(close, length=30, fast=2, slow=30)

    # Volume confirmation: current volume > 1.5 x 30-day average
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if data is not ready
        if (np.isnan(dkama[i]) or np.isnan(wkama_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above daily KAMA with volume spike and weekly uptrend
            if close[i] > dkama[i] and close[i-1] <= dkama[i-1] and volume_spike[i] and close[i] > wkama_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below daily KAMA with volume spike and weekly downtrend
            elif close[i] < dkama[i] and close[i-1] >= dkama[i-1] and volume_spike[i] and close[i] < wkama_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below daily KAMA or weekly trend turns down
            if close[i] < dkama[i] and close[i-1] >= dkama[i-1] or close[i] < wkama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above daily KAMA or weekly trend turns up
            if close[i] > dkama[i] and close[i-1] <= dkama[i-1] or close[i] > wkama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals