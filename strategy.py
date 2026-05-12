#!/usr/bin/env python3

# 12h_1D_Camarilla_R1S1_Breakout_Trend_Filter
# Hypothesis: Long when price breaks above Camarilla R1 on 12h with 1d uptrend and volume confirmation; short when price breaks below S1 with 1d downtrend and volume confirmation. Uses Camarilla pivot levels from daily timeframe for structure, 12h for execution, and filters by 1d EMA trend and volume spike to avoid false breakouts. Designed for 12-37 trades/year to stay within fee limits.

name = "12h_1D_Camarilla_R1S1_Breakout_Trend_Filter"
timeframe = "12h"
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

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Typical price for pivot
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12

    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # 1d EMA for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Volume confirmation: current volume > 1.5x average of last 10 periods
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(10, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 1d
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Break above R1 with bullish trend and volume confirmation
            if close[i] > r1_aligned[i] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with bearish trend and volume confirmation
            elif close[i] < s1_aligned[i] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or trend turns bearish
            if close[i] < r1_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or trend turns bullish
            if close[i] > s1_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals