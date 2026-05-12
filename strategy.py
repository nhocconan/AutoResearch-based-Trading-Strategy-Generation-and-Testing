#!/usr/bin/env python3
# 6h_1W_1D_Camarilla_R4S4_Breakout_TrendFilter_Volume
# Hypothesis: Breakouts at weekly Camarilla R4/S4 levels (stronger levels) with daily trend filter and volume confirmation.
# Uses weekly levels for stronger support/resistance, reducing false breaks. Works in bull/bear: buy R4 breaks in uptrend, sell S4 breaks in downtrend.
# Volume confirmation ensures breakout validity. Designed for 6h to limit trade frequency (target: 50-150 total trades over 4 years).

name = "6h_1W_1D_Camarilla_R4S4_Breakout_TrendFilter_Volume"
timeframe = "6h"
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

    # Get weekly data for stronger Camarilla levels (R4/S4)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate Camarilla levels from previous weekly OHLC (R4 and S4)
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values

    # Camarilla R4 and S4 levels (outer bands)
    camarilla_r4 = prev_weekly_close + (prev_weekly_high - prev_weekly_low) * 1.1 / 2
    camarilla_s4 = prev_weekly_close - (prev_weekly_high - prev_weekly_low) * 1.1 / 2

    # Align weekly Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)

    # Get daily data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume confirmation: current volume > 2.0x average of last 20 periods (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]

        if position == 0:
            # LONG: Break above weekly Camarilla R4 in uptrend with volume confirmation
            if (close[i] > camarilla_r4_aligned[i] and uptrend and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly Camarilla S4 in downtrend with volume confirmation
            elif (close[i] < camarilla_s4_aligned[i] and downtrend and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below weekly R4 or trend reversal
            if close[i] < camarilla_r4_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above weekly S4 or trend reversal
            if close[i] > camarilla_s4_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals