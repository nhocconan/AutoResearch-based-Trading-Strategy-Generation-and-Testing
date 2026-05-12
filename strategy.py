# 12h_1d_Camarilla_R1_S1_Breakout
# Hypothesis: Camarilla pivot levels (R1/S1) on daily chart act as key support/resistance.
# Breakout above R1 with 1w uptrend (EMA200) and volume confirmation = long.
# Breakdown below S1 with 1w downtrend = short.
# Uses daily pivot for structure, weekly trend filter to avoid counter-trend trades.
# Target: 15-30 trades/year per symbol with disciplined risk management.
# Works in bull markets (breakouts catch momentum) and bear markets (breakdowns catch drops).

name = "12h_1d_Camarilla_R1_S1_Breakout"
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

    # Get 1d data for Camarilla pivot calculation (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels for each 1d bar
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    cam_R1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    cam_S1 = close_1d - 1.1 * (high_1d - low_1d) / 12

    # Align Camarilla levels to 12h timeframe (wait for daily close)
    cam_R1_aligned = align_htf_to_ltf(prices, df_1d, cam_R1)
    cam_S1_aligned = align_htf_to_ltf(prices, df_1d, cam_S1)

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # 1w EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(cam_R1_aligned[i]) or np.isnan(cam_S1_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above Camarilla R1 + 1w uptrend + volume confirmation
            if close[i] > cam_R1_aligned[i] and close[i] > ema200_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Camarilla S1 + 1w downtrend + volume confirmation
            elif close[i] < cam_S1_aligned[i] and close[i] < ema200_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below Camarilla S1 or 1w trend turns down
            if close[i] < cam_S1_aligned[i] or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above Camarilla R1 or 1w trend turns up
            if close[i] > cam_R1_aligned[i] or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals