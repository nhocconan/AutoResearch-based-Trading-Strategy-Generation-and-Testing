#!/usr/bin/env python3
#160137: 4h_Donchian_Breakout_20_1wTrend_VolumeConfirm
# Hypothesis: Price breaking above/below 4h Donchian(20) with weekly EMA10 trend filter and volume confirmation captures strong trending moves while avoiding false breakouts. Weekly trend filter ensures alignment with higher timeframe direction, working in both bull and bear markets. Uses 4h timeframe with weekly EMA10 trend filter for higher timeframe context.

name = "4h_Donchian_Breakout_20_1wTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate weekly EMA10 trend filter
    ema_10_1w = pd.Series(df_1w['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)

    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after Donchian warmup
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_10_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian + weekly uptrend + volume confirmation
            if (close[i] > high_roll[i] and 
                close[i] > ema_10_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below lower Donchian + weekly downtrend + volume confirmation
            elif (close[i] < low_roll[i] and 
                  close[i] < ema_10_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below lower Donchian (trend reversal)
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price closes above upper Donchian (trend reversal)
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals