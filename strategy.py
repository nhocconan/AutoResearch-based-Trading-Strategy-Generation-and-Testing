#!/usr/bin/env python3
# 1H_Camarilla_R1S1_Breakout_4hTrend_Volume
# Hypothesis: Breakouts at daily Camarilla R1/S1 levels with 4h EMA trend filter and volume confirmation, using 1h for entry timing.
# Works in bull/bear markets: In uptrends, buy R1 breakouts; in downtrends, sell S1 breakdowns. Volume reduces false signals.
# Uses 4h trend for direction and 1h for precise entry, targeting 15-37 trades/year to avoid fee drag.

name = "1H_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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

    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)

    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate Camarilla levels from previous 1d OHLC (avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    # Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12

    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Volume confirmation: current volume > 1.5x average of last 24 periods (1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]

        if position == 0:
            # LONG: Break above Camarilla R1 in uptrend with volume confirmation
            if (close[i] > camarilla_r1_aligned[i] and uptrend and volume_ok[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Break below Camarilla S1 in downtrend with volume confirmation
            elif (close[i] < camarilla_s1_aligned[i] and downtrend and volume_ok[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Camarilla range (below R1) or trend reversal
            if close[i] < camarilla_r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters Camarilla range (above S1) or trend reversal
            if close[i] > camarilla_s1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals