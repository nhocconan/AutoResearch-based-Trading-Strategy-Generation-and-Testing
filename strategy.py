#!/usr/bin/env python3
# 6h_Elder_Ray_Power_1wTrend_Volume
# Hypothesis: Elder Ray (Bull/Bear Power) from 1d combined with weekly trend filter and volume spikes
# identifies strong momentum moves in both bull and bear markets. Bull Power > 0 indicates bullish
# momentum (price above EMA), Bear Power < 0 indicates bearish momentum (price below EMA).
# Weekly trend ensures we trade with the higher timeframe direction, reducing whipsaws.
# Volume spikes confirm institutional participation. Works in all market regimes by adapting
# to the prevailing weekly trend while using 1d for precise entry/exit.

name = "6h_Elder_Ray_Power_1wTrend_Volume"
timeframe = "6h"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)

    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)

    # Calculate EMA13 for Elder Ray (standard setting)
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Calculate Bull Power and Bear Power
    bull_power = df_1d['high'].values - ema_13_1d  # High - EMA
    bear_power = df_1d['low'].values - ema_13_1d   # Low - EMA

    # Align Elder Ray to 6h timeframe (needs only completed daily candle)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)

    # Weekly EMA34 trend filter (only needs completed weekly candle)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Volume confirmation: current volume > 2.0x average of last 50 periods
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]

        if position == 0:
            # LONG: Bull Power > 0 (bullish momentum) + weekly uptrend + volume spike
            if (bull_power_aligned[i] > 0 and 
                weekly_uptrend and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (bearish momentum) + weekly downtrend + volume spike
            elif (bear_power_aligned[i] < 0 and 
                  weekly_downtrend and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power >= 0 (loss of bullish momentum) or weekly trend turns down
            if (bear_power_aligned[i] >= 0) or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power <= 0 (loss of bearish momentum) or weekly trend turns up
            if (bull_power_aligned[i] <= 0) or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals