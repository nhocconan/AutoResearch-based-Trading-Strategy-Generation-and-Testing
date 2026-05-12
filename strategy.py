#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Trade Camarilla pivot breakouts on 12h with 1d EMA trend filter and volume confirmation. Works in both bull and bear markets by requiring higher timeframe trend alignment and volume confirmation to reduce false signals in choppy periods. Target: 20-40 trades/year.
"""

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

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter (call once before loop)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)

    # Calculate Camarilla pivot levels from previous 1d high/low/close
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values

    # Pivot = (H+L+C)/3
    pivot = (high_daily + low_daily + close_daily) / 3.0
    # Range = H - L
    range_daily = high_daily - low_daily
    # R1 = C + (H-L)*1.1/12
    r1 = close_daily + range_daily * 1.1 / 12
    # S1 = C - (H-L)*1.1/12
    s1 = close_daily - range_daily * 1.1 / 12

    # Align to 12h timeframe (previous day's levels available at open)
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema34_val = ema34_daily_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R1 + daily uptrend + volume confirmation
            if close[i] > r1_val and close[i] > ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S1 + daily downtrend + volume confirmation
            elif close[i] < s1_val and close[i] < ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or close below daily EMA34
            if close[i] < s1_val or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or close above daily EMA34
            if close[i] > r1_val or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals