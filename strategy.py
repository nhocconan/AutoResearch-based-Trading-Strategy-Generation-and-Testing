#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12h_EMA21_Trend_Volume
# Hypothesis: Price breaking above/below daily Camarilla R1/S1 levels with 12h EMA21 trend filter and volume confirmation captures strong trending moves. Works in bull/bear by following higher timeframe trend. Uses 4h timeframe with daily Camarilla levels and 12h EMA21 trend filter. Designed to avoid false breakouts and reduce trade frequency compared to R3/S3 variants.

name = "4h_Camarilla_R1_S1_Breakout_12h_EMA21_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily Camarilla levels: R1, S1
    # R1 = close + 1.1 * (high - low) / 4
    # S1 = close - 1.1 * (high - low) / 4
    camarilla_range = high_1d - low_1d
    r1_level = close_1d + 1.1 * camarilla_range / 4
    s1_level = close_1d - 1.1 * camarilla_range / 4

    # Align Camarilla levels to 4h timeframe
    r1_level_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_level_aligned = align_htf_to_ltf(prices, df_1d, s1_level)

    # Get 12h data ONCE before loop for EMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values

    # 12h EMA21 trend filter
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(21, n):  # Start after EMA21 warmup
        if (np.isnan(r1_level_aligned[i]) or np.isnan(s1_level_aligned[i]) or 
            np.isnan(ema_21_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + 12h EMA21 uptrend + volume confirmation
            if (close[i] > r1_level_aligned[i] and 
                close[i] > ema_21_12h_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + 12h EMA21 downtrend + volume confirmation
            elif (close[i] < s1_level_aligned[i] and 
                  close[i] < ema_21_12h_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 12h EMA21 (trend reversal)
            if close[i] < ema_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 12h EMA21 (trend reversal)
            if close[i] > ema_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals