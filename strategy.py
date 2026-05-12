#!/usr/bin/env python3
# 1h_4H1D_Camarilla_R1S1_Breakout_Volume
# Hypothesis: Use 4h/1d Camarilla pivot levels as structural support/resistance with 1h breakout entries.
# Only trade in direction of 1d trend (EMA50) and require volume spike. This captures institutional
# interest at key levels while filtering noise. Designed for 1h timeframe to balance trade frequency
# and signal quality, targeting 15-35 trades/year. Works in bull/bear by following 1d trend.

name = "1h_4H1D_Camarilla_R1S1_Breakout_Volume"
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

    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')

    # 4h Camarilla levels (based on previous day's range)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    camarilla_range = (high_4h - low_4h) * 1.1 / 12
    r1_4h = close_4h + camarilla_range
    s1_4h = close_4h - camarilla_range

    # Align 4h Camarilla levels to 1h
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)

    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume spike: current > 2.0x average of last 24 bars
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + 1d EMA50 uptrend + volume spike
            if (close[i] > r1_4h_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + 1d EMA50 downtrend + volume spike
            elif (close[i] < s1_4h_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 (reversion to mean) or trend fails
            if close[i] < s1_4h_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above R1 (reversion to mean) or trend fails
            if close[i] > r1_4h_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals