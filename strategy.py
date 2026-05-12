#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R1, S1) from daily data act as strong support/resistance.
# Breakout above R1 with 1d uptrend and volume confirmation signals long.
# Breakdown below S1 with 1d downtrend and volume confirmation signals short.
# Uses 12h timeframe to limit trades and reduce fee drag. Works in bull via R1 breakouts and bear via S1 breakdowns.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate Camarilla levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low

    # Camarilla R1 and S1
    r1 = prev_close + (prev_range * 1.1 / 12)
    s1 = prev_close - (prev_range * 1.1 / 12)

    # Align Camarilla levels to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # Daily trend filter: EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema34 = ema_34_1d_aligned[i]
        vol_ok = volume_ok[i]

        if position == 0:
            # LONG: Break above R1 with uptrend and volume
            if price > r1_level and price > ema34 and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with downtrend and volume
            elif price < s1_level and price < ema34 and vol_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns below R1 or trend turns down
            if price < r1_level or price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns above S1 or trend turns up
            if price > s1_level or price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals