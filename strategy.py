#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: On 12h timeframe, trade breakouts of daily Camarilla R1/S1 levels in the direction of the 1-day EMA34 trend, confirmed by volume spike (>1.8x 20-period average). In bull markets, buy breakouts above R1; in bear markets, sell breakdowns below S1. Uses EMA34 for trend filter and volume confirmation to ensure institutional participation. Designed for 12-37 trades/year on 12h timeframe to minimize fee drag. Works in both bull and bear by capturing trend-aligned breakouts with volume confirmation.

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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

    # Calculate EMA34 on daily timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = np.zeros_like(close_1d)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 32) / 34
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Calculate daily Camarilla levels (R1, S1) from previous day
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    R1 = np.zeros_like(close_1d)
    S1 = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i >= 1:
            R1[i] = close_1d[i-1] + 1.1 * (high_1d[i-1] - low_1d[i-1]) / 12
            S1[i] = close_1d[i-1] - 1.1 * (high_1d[i-1] - low_1d[i-1]) / 12
        else:
            R1[i] = close_1d[0]
            S1[i] = close_1d[0]
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)

    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if EMA or Camarilla data is not ready
        if i < 20 or np.isnan(ema_34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 with volume spike and price > EMA34 (uptrend)
            if close[i] > R1_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 with volume spike and price < EMA34 (downtrend)
            elif close[i] < S1_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below S1 (reversal signal)
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above R1 (reversal signal)
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals