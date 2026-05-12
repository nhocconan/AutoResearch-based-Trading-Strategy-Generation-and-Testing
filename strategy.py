#!/usr/bin/env python3
# 1D_1W_Camarilla_R4S4_Breakout_TrendFilter_Volume
# Hypothesis: Breakouts at weekly Camarilla R4/S4 levels with 1d EMA100 trend filter and volume confirmation.
# Uses weekly levels to capture stronger support/resistance, daily trend filter for bias, volume for confirmation.
# Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year). Works in bull/bear markets.
# Weekly levels reduce noise and false breakouts compared to daily levels.

name = "1D_1W_Camarilla_R4S4_Breakout_TrendFilter_Volume"
timeframe = "1d"
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

    # Get weekly data for Camarilla R4/S4 levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly Camarilla levels from previous week
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values

    # Camarilla R4 and S4 levels (outer bands)
    camarilla_r4 = prev_week_close + (prev_week_high - prev_week_low) * 1.1 / 2
    camarilla_s4 = prev_week_close - (prev_week_high - prev_week_low) * 1.1 / 2

    # Align weekly levels to daily timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)

    # Get daily data for EMA100 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Daily EMA100 trend filter
    ema_100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)

    # Volume confirmation: current volume > 1.5x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_100_1d_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA100
        uptrend = close[i] > ema_100_1d_aligned[i]
        downtrend = close[i] < ema_100_1d_aligned[i]

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
            # EXIT LONG: Price re-enters below R4 or trend reversal
            if close[i] < camarilla_r4_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S4 or trend reversal
            if close[i] > camarilla_s4_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals