#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# 160108: 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Price breaking above/below 12-hour Camarilla R1/S1 levels with weekly trend filter (1w EMA50) and volume confirmation captures strong trending moves while avoiding false breakouts. Works in bull/bear by following the higher timeframe trend direction. Targets 50-150 total trades over 4 years with low turnover to minimize fee drag.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate 1-week high, low, close for weekly trend (EMA50)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # 1-week EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate 1-day high, low, close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels: R1, S1
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_range = high_1d - low_1d
    r1_level = close_1d + 1.1 * camarilla_range / 12
    s1_level = close_1d - 1.1 * camarilla_range / 12

    # Align Camarilla levels to 12h timeframe
    r1_level_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_level_aligned = align_htf_to_ltf(prices, df_1d, s1_level)

    # Volume confirmation: >1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Start after EMA50 warmup
        if (np.isnan(r1_level_aligned[i]) or np.isnan(s1_level_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + 1w EMA50 uptrend + volume confirmation
            if (close[i] > r1_level_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + 1w EMA50 downtrend + volume confirmation
            elif (close[i] < s1_level_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1w EMA50 (trend reversal)
            if close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 1w EMA50 (trend reversal)
            if close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals