#!/usr/bin/env python3
"""
12h_1D_Camarilla_R1_S1_Breakout_Trend_With_Volume_Confirmation
Hypothesis: Camarilla R1/S1 levels from 1d provide high-probability reversal zones.
Price breaking above R1 (resistance) with 1d volume > 1.5x 20-period average triggers long,
breaking below S1 (support) with volume surge triggers short.
Trend filter: price above/below 1d EMA34 avoids counter-trend trades.
Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
Works in bull (breakouts continue) and bear (fades at S1/R1) markets.
"""

name = "12h_1D_Camarilla_R1_S1_Breakout_Trend_With_Volume_Confirmation"
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

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    # Calculate 1d Camarilla levels (based on previous day)
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Previous day values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day has no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan

    # Camarilla levels
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12

    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values

    # Align to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)

    # 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        ema34 = ema_34_aligned[i]
        vol_1d = volume_1d_aligned[i]
        vol_avg = vol_avg_20_1d_aligned[i]

        if np.isnan(r1) or np.isnan(s1) or np.isnan(ema34) or np.isnan(vol_1d) or np.isnan(vol_avg):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + volume surge + above EMA34 (uptrend)
            if close[i] > r1 and vol_1d > vol_avg * 1.5 and close[i] > ema34:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume surge + below EMA34 (downtrend)
            elif close[i] < s1 and vol_1d > vol_avg * 1.5 and close[i] < ema34:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R1 or below EMA34 (trend change)
            if close[i] < r1 or close[i] < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S1 or above EMA34 (trend change)
            if close[i] > s1 or close[i] > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals