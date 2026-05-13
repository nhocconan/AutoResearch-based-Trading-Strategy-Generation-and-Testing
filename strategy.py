#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: Camarilla pivot levels on 1h timeframe with 4h EMA trend filter and 1d volume spike confirmation.
# Long when price breaks above R1 level with volume spike and above 4h EMA50 (uptrend).
# Short when price breaks below S1 level with volume spike and below 4h EMA50 (downtrend).
# Exit when price crosses back through pivot point (mean reversion to equilibrium).
# Uses 1h for entry timing, 4h for trend direction, 1d for volume confirmation.
# Designed for 15-30 trades/year to minimize fee drag while capturing institutional breakouts.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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

    # Camarilla pivot levels for 1h timeframe
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # Pivot = (high + low + close)/3
    hl_range = high - low
    r1 = close + (1.1 * hl_range / 12)
    s1 = close - (1.1 * hl_range / 12)
    pivot = (high + low + close) / 3

    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)

    # 1d volume confirmation: current volume > 2.0 x 24-period average
    vol_ma_1d = np.zeros_like(volume)
    for i in range(24, n):
        vol_ma_1d[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (2.0 * vol_ma_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if data not ready
        if np.isnan(ema_4h_aligned[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 with volume spike and above 4h EMA50 (uptrend)
            if close[i] > r1[i] and volume_spike[i] and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 with volume spike and below 4h EMA50 (downtrend)
            elif close[i] < s1[i] and volume_spike[i] and close[i] < ema_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (mean reversion)
            if close[i] < pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (mean reversion)
            if close[i] > pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals