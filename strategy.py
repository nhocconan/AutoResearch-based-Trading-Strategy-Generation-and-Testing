#!/usr/bin/env python3
# 4h_1D_Camarilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: Breakouts at daily Camarilla R1/S1 levels with 1-day EMA trend filter and volume confirmation.
# Targets tight entries (15-25 trades/year) to avoid fee drag. Works in bull/bear markets:
# - Buy when price breaks above R1 in bullish 1-day EMA trend with volume spike
# - Sell when price breaks below S1 in bearish 1-day EMA trend with volume spike
# Uses tight stops: exit when price re-enters the CPR (central pivot range) or trend fails.

name = "4h_1D_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_signal = ema_34  # Above EMA = bullish trend
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_signal)

    # Get 1d data for Camarilla R1/S1 levels (from previous day)
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate Camarilla levels from previous 1d OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low

    # Camarilla R1 and S1 levels (inner bands)
    camarilla_r1 = close + range_hl * 1.1 / 12
    camarilla_s1 = close - range_hl * 1.1 / 12

    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 1-day EMA34
        bullish_trend = close[i] > ema_34_aligned[i]
        bearish_trend = close[i] < ema_34_aligned[i]

        if position == 0:
            # LONG: Break above Camarilla R1 in bullish trend with volume confirmation
            if (close[i] > camarilla_r1_aligned[i] and bullish_trend and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S1 in bearish trend with volume confirmation
            elif (close[i] < camarilla_s1_aligned[i] and bearish_trend and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or trend turns bearish
            if close[i] < camarilla_r1_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or trend turns bullish
            if close[i] > camarilla_s1_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals