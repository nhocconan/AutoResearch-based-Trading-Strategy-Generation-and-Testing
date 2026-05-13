# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# 4h_VolumeSpike_InsideBar_Breakout
# Hypothesis: Breakout of inside bar (IB) with volume spike (>2x 20-bar avg) and price > 200-period SMA (trend filter). Works in bull/bear by requiring trend alignment. Inside bar provides low-risk entry, volume confirms breakout strength. Target ~25 trades/year on 4h to minimize fee drag.

name = "4h_VolumeSpike_InsideBar_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # 200-period SMA for trend filter (using close)
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values

    # Inside bar detection: current high <= previous high AND current low >= previous low
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    inside_bar = (high <= prev_high) & (low >= prev_low)
    inside_bar[0] = False  # first bar has no previous

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required value is NaN
        if np.isnan(sma200[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Inside bar breakout up + volume spike + price > SMA200
            if (inside_bar[i] and 
                high[i] > prev_high[i] and  # break above inside bar high
                volume[i] > vol_avg_20[i] * 2.0 and
                close[i] > sma200[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Inside bar breakout down + volume spike + price < SMA200
            elif (inside_bar[i] and 
                  low[i] < prev_low[i] and  # break below inside bar low
                  volume[i] > vol_avg_20[i] * 2.0 and
                  close[i] < sma200[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below inside bar low (trailing stop)
            if close[i] < prev_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above inside bar high
            if close[i] > prev_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals