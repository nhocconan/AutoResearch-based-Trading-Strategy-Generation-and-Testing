#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R1 with 1d uptrend and volume spike.
# Short when price breaks below S1 with 1d downtrend and volume spike.
# Works in bull markets via R1 breakouts and in bear markets via S1 breakdowns.
# Uses daily EMA34 for trend filter and volume > 1.5x 20-period average for confirmation.
# Designed for 12h timeframe to limit trades (target: 50-150 total over 4 years).

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

    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    # Daily close for EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_prev = np.roll(close_1d, 1)
    high_prev = np.roll(df_1d['high'].values, 1)
    low_prev = np.roll(df_1d['low'].values, 1)
    close_prev[0] = close_1d[0]  # avoid NaN on first bar
    high_prev[0] = df_1d['high'].values[0]
    low_prev[0] = df_1d['low'].values[0]

    r1 = close_prev + (high_prev - low_prev) * 1.1 / 12
    s1 = close_prev - (high_prev - low_prev) * 1.1 / 12

    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

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

        # Trend filter from daily EMA34
        price_above_daily_ema = close[i] > ema_34_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_34_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above R1, daily uptrend, volume confirmation
            if close[i] > r1_aligned[i] and price_above_daily_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1, daily downtrend, volume confirmation
            elif close[i] < s1_aligned[i] and price_below_daily_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or daily trend turns down
            if close[i] < s1_aligned[i] or not price_above_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or daily trend turns up
            if close[i] > r1_aligned[i] or not price_below_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals