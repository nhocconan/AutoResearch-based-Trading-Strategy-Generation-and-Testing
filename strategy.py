#!/usr/bin/env python3
"""
4h_ThreeBarBreakout_1dTrend_Volume
Hypothesis: Enter on 3-bar breakouts (close > prior 3-bar high or < prior 3-bar low) aligned with 1d EMA50 trend and volume spike. Exit on reversal of 3-bar pattern. This targets ~30 trades/year by requiring trend alignment, volume confirmation, and momentum confirmation. Works in bull/bear markets via trend-following entries and momentum-based exits.
Timeframe: 4h
"""

name = "4h_ThreeBarBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for EMA50 trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Three-bar breakout levels: lookback 3 bars
    high_3 = pd.Series(high).rolling(window=3, min_periods=3).max().shift(1).values
    low_3 = pd.Series(low).rolling(window=3, min_periods=3).min().shift(1).values

    # Volume spike: current > 2.0x average of last 6 bars (1 day on 4h)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(80, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_3[i]) or np.isnan(low_3[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: close > 3-bar high + price > 1d EMA50 + volume spike
            if (close[i] > high_3[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: close < 3-bar low + price < 1d EMA50 + volume spike
            elif (close[i] < low_3[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close < 3-bar low (momentum reversal)
            if close[i] < low_3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close > 3-bar high (momentum reversal)
            if close[i] > high_3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals