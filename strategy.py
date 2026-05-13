#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) on 1d timeframe provide key support/resistance levels.
# Long when price breaks above R1 with volume spike and price above 1d EMA34 (uptrend).
# Short when price breaks below S1 with volume spike and price below 1d EMA34 (downtrend).
# Exit when price crosses back through the pivot point (mean reversion within day's range).
# Uses 1d timeframe for Camarilla levels and EMA trend filter to avoid whipsaws.
# Designed for 20-40 trades/year to minimize fee drift. Works in both bull and bear by
# capturing institutional reversal points with trend alignment.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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

    # Camarilla pivot levels from previous 1d
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels: R1, S1, PP
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    PP = np.full(n, np.nan)
    for i in range(len(close_1d)):
        # Previous day's range
        range_1d = high_1d[i] - low_1d[i]
        # Camarilla formulas
        R1_val = close_1d[i] + range_1d * 1.1 / 12
        S1_val = close_1d[i] - range_1d * 1.1 / 12
        PP_val = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
        R1[i] = R1_val
        S1[i] = S1_val
        PP[i] = PP_val

    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)

    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume confirmation: current volume > 2.0 x 24-period average (6h)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(PP_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R1 with volume spike and price above 1d EMA34 (uptrend)
            if close[i] > R1_aligned[i] and volume_spike[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume spike and price below 1d EMA34 (downtrend)
            elif close[i] < S1_aligned[i] and volume_spike[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below pivot point (mean reversion)
            if close[i] < PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above pivot point
            if close[i] > PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals