#!/usr/bin/env python3
"""
1h_4h_1d_Trend_Follow_with_Volume_and_Time_Filter
Hypothesis: Trend following strategy using 4h EMA for direction, 1d ATR for volatility filter, and volume spike for confirmation on 1h timeframe. Restricts trading to 08-20 UTC to avoid low-liquidity periods. Designed to work in both bull and bear markets by following established trends with proper risk controls.
"""

name = "1h_4h_1d_Trend_Follow_with_Volume_and_Time_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')

    # 4h EMA21 for trend direction
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)

    # 1d ATR14 for volatility filter (requires 2 extra bars for confirmation)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d, additional_delay_bars=2)

    # Volume spike: >2.0x 24-period average (1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    # Time filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    time_filter = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or not time_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above 4h EMA21 + ATR filter + volume spike
            if (close[i] > ema_21_4h_aligned[i] and 
                close[i] > ema_21_4h_aligned[i] + 0.5 * atr_14_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price below 4h EMA21 - ATR filter + volume spike
            elif (close[i] < ema_21_4h_aligned[i] and 
                  close[i] < ema_21_4h_aligned[i] - 0.5 * atr_14_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 4h EMA21
            if close[i] < ema_21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above 4h EMA21
            if close[i] > ema_21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals