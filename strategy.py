#!/usr/bin/env python3
# 4h_1D_Camarilla_R1_S1_Breakout_Volume
# Hypothesis: Breakouts at daily Camarilla R1/S1 levels with volume confirmation on 4h timeframe.
# Uses 1d timeframe for Camarilla levels and momentum confirmation, 4h for entry/exit.
# Designed to work in both bull and bear markets by requiring volume confirmation and momentum alignment.
# Targets 20-50 trades/year on 4h timeframe to avoid fee drag.

name = "4h_1D_Camarilla_R1_S1_Breakout_Volume"
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

    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate 1d EMA for momentum filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate Camarilla R1 and S1 levels from previous 1d OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12

    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Momentum filter: price above/below 34-period EMA on 1d
        bullish_momentum = close[i] > ema_1d_aligned[i]
        bearish_momentum = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Break above Camarilla R1 with bullish momentum and volume confirmation
            if (close[i] > camarilla_r1_aligned[i] and bullish_momentum and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S1 with bearish momentum and volume confirmation
            elif (close[i] < camarilla_s1_aligned[i] and bearish_momentum and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or momentum turns bearish
            if close[i] < camarilla_r1_aligned[i] or not bullish_momentum:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or momentum turns bullish
            if close[i] > camarilla_s1_aligned[i] or not bearish_momentum:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals