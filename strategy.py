#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend
# Hypothesis: Breakout above Camarilla R1 or below S1 with volume spike and 1d EMA trend filter.
# Camarilla levels from daily chart provide institutional support/resistance.
# Breakouts with volume confirm institutional interest. Trend filter avoids counter-trend trades.
# Works in bull/bear by following 1d EMA trend. Target: 20-50 trades/year.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend"
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

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels for each day
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_R1 = close_1d + range_1d * 1.1 / 12
    camarilla_S1 = close_1d - range_1d * 1.1 / 12

    # Forward fill to get today's levels
    camarilla_R1_ff = np.full(len(df_1d), np.nan)
    camarilla_S1_ff = np.full(len(df_1d), np.nan)
    last_R1 = np.nan
    last_S1 = np.nan
    for i in range(len(df_1d)):
        if not np.isnan(camarilla_R1[i]):
            last_R1 = camarilla_R1[i]
        if not np.isnan(camarilla_S1[i]):
            last_S1 = camarilla_S1[i]
        camarilla_R1_ff[i] = last_R1
        camarilla_S1_ff[i] = last_S1

    # Align Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1_ff)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1_ff)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R1 with volume spike and 1d EMA34 uptrend
            if close[i] > camarilla_R1_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume spike and 1d EMA34 downtrend
            elif close[i] < camarilla_S1_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below R1 (failed breakout)
            if close[i] < camarilla_R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above S1 (failed breakout)
            if close[i] > camarilla_S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals