#!/usr/bin/env python3

# 4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike
# Hypothesis: Buy when price breaks above daily Camarilla R1 with 4h EMA50 uptrend and volume spike, sell when breaks below S1 with downtrend and volume spike.
# Uses 4h timeframe for entries, daily for Camarilla levels and EMA trend filter, weekly for higher timeframe trend confirmation.
# Volume spike filters breakouts. Designed for 20-50 trades/year to minimize fee drag and work in both bull and bear markets via trend alignment.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Get weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate daily Camarilla levels (based on previous day)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    rang = prev_high - prev_low
    R1 = prev_close + rang * 1.1 / 12
    S1 = prev_close - rang * 1.1 / 12
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)

    # Volume confirmation: current volume > 2.0x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Higher timeframe trend filter (weekly EMA34)
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        weekly_downtrend = close[i] < ema_1w_aligned[i]

        if position == 0:
            # LONG: Price crosses above R1 with daily EMA34 uptrend, weekly uptrend, and volume spike
            if (close[i] > R1_aligned[i] and close[i-1] <= R1_aligned[i-1] and
                close[i] > ema_1d_aligned[i] and weekly_uptrend and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below S1 with daily EMA34 downtrend, weekly downtrend, and volume spike
            elif (close[i] < S1_aligned[i] and close[i-1] >= S1_aligned[i-1] and
                  close[i] < ema_1d_aligned[i] and weekly_downtrend and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 or daily EMA34 turns down
            if (close[i] < S1_aligned[i] and close[i-1] >= S1_aligned[i-1]) or (close[i] < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 or daily EMA34 turns up
            if (close[i] > R1_aligned[i] and close[i-1] <= R1_aligned[i-1]) or (close[i] > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals