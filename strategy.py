#!/usr/bin/env python3
# 4h_1D_Camarilla_R1S1_Breakout_1wTrend_Volume
# Hypothesis: Breakouts at daily Camarilla R1/S1 levels with weekly EMA trend filter and volume confirmation.
# Weekly trend ensures alignment with long-term bias, reducing counter-trend trades. Works in bull/bear markets:
# In uptrends (weekly EMA up), buy R1 breakouts; in downtrends (weekly EMA down), sell S1 breakdowns.
# Volume confirms breakout strength. Designed for 4h to balance trade frequency and signal quality.

name = "4h_1D_Camarilla_R1S1_Breakout_1wTrend_Volume"
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

    # Get 4h data for EMA trend filter (optional confirmation)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    # 4h EMA20 trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)

    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate Camarilla levels from previous 1d OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    # Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12

    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filters
        uptrend_4h = close[i] > ema_20_4h_aligned[i]
        uptrend_1w = close[i] > ema_50_1w_aligned[i]
        downtrend_4h = close[i] < ema_20_4h_aligned[i]
        downtrend_1w = close[i] < ema_50_1w_aligned[i]

        if position == 0:
            # LONG: Break above Camarilla R1 in uptrend (4h and 1w) with volume confirmation
            if (close[i] > camarilla_r1_aligned[i] and uptrend_4h and uptrend_1w and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S1 in downtrend (4h and 1w) with volume confirmation
            elif (close[i] < camarilla_s1_aligned[i] and downtrend_4h and downtrend_1w and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Camarilla range (below R1) or trend reversal
            if close[i] < camarilla_r1_aligned[i] or not (uptrend_4h and uptrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Camarilla range (above S1) or trend reversal
            if close[i] > camarilla_s1_aligned[i] or not (downtrend_4h and downtrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals