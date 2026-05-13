#!/usr/bin/env python3
# 6h_WeeklyPivot_CamarillaBreakout_1dTrend_Volume
# Hypothesis: Breakout from daily Camarilla pivot levels (R4/S4) with weekly pivot direction and volume confirmation.
# Uses weekly pivot trend to filter direction and avoid counter-trend whipsaws.
# Works in bull/bear markets by aligning with higher timeframe structure.
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "6h_WeeklyPivot_CamarillaBreakout_1dTrend_Volume"
timeframe = "6h"
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

    # Get weekly data for pivot trend
    df_1w = get_htf_data(prices, '1w')
    # Weekly trend: close above/below 20-period EMA
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels from previous day
    # R4 = close + 1.5 * (high - low)
    # S4 = close - 1.5 * (high - low)
    r4 = df_1d['close'] + 1.5 * (df_1d['high'] - df_1d['low'])
    s4 = df_1d['close'] - 1.5 * (df_1d['high'] - df_1d['low'])
    # Align to 6h timeframe (use previous day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R4 in weekly uptrend with volume spike
            if close[i] > r4_aligned[i] and close[i] > ema_20_1w_aligned[i]:
                if volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Break below S4 in weekly downtrend with volume spike
            elif close[i] < s4_aligned[i] and close[i] < ema_20_1w_aligned[i]:
                if volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S4 or weekly trend turns down
            if close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R4 or weekly trend turns up
            if close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals