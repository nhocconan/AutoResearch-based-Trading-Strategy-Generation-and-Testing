#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrend_1dVolumeSpike
Hypothesis: 12h timeframe reduces trade frequency while maintaining signal quality. 
Long when price breaks above weekly Camarilla R3 + weekly uptrend + daily volume spike.
Short when price breaks below weekly Camarilla S3 + weekly downtrend + daily volume spike.
Weekly trend filter provides robustness across market regimes, reducing whipsaws.
"""

name = "12h_Camarilla_R3S3_Breakout_1wTrend_1dVolumeSpike"
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

    # Get weekly data for trend and Camarilla levels (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values

    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Weekly Camarilla pivot levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_R3_1w = close_1w + (high_1w - low_1w) * 1.1 / 4
    camarilla_S3_1w = close_1w - (high_1w - low_1w) * 1.1 / 4

    # Align weekly Camarilla levels to 12h timeframe
    camarilla_R3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R3_1w)
    camarilla_S3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S3_1w)

    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(camarilla_R3_1w_aligned[i]) or np.isnan(camarilla_S3_1w_aligned[i]) or \
           np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above weekly Camarilla R3 + weekly uptrend + daily volume spike
            if close[i] > camarilla_R3_1w_aligned[i] and close[i] > ema20_1w_aligned[i] and volume[i] > vol_avg_20_1d_aligned[i] * 2:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below weekly Camarilla S3 + weekly downtrend + daily volume spike
            elif close[i] < camarilla_S3_1w_aligned[i] and close[i] < ema20_1w_aligned[i] and volume[i] > vol_avg_20_1d_aligned[i] * 2:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below weekly Camarilla S3 or weekly trend turns down
            if close[i] < camarilla_S3_1w_aligned[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above weekly Camarilla R3 or weekly trend turns up
            if close[i] > camarilla_R3_1w_aligned[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals