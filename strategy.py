#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: 4H Camarilla R1/S1 breakout with daily EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R1 with daily uptrend and volume spike.
# Short when price breaks below S1 with daily downtrend and volume spike.
# Uses volume filter to ensure institutional participation and reduce false signals.
# Target: 20-40 trades/year to minimize fee drag. Works in bull/bear by trading breakouts with trend alignment.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
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

    # Calculate daily Camarilla levels (using prior day's range)
    df_1d = get_htf_data(prices, '1d')
    # Camarilla levels: based on previous day's high, low, close
    h1 = df_1d['high']
    l1 = df_1d['low']
    c1 = df_1d['close']
    r1 = c1 + (h1 - l1) * 1.1 / 12
    s1 = c1 - (h1 - l1) * 1.1 / 12
    # Align to 4h timeframe (wait for daily candle to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)

    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 with daily uptrend and volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_34_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 with daily downtrend and volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_34_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below S1 or trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above R1 or trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals