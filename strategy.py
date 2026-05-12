#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Camarilla R1/S1 breakouts with weekly trend filter and volume confirmation capture institutional breakouts in both bull and bear markets.
Breakouts above R1 + weekly uptrend = long; breakdowns below S1 + weekly downtrend = short.
Uses weekly EMA for trend filter to avoid whipsaws and capture major moves.
Target: 15-30 trades/year per symbol with disciplined risk management.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate daily pivot points for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1 = C + (H-L)*1.12/12, S1 = C - (H-L)*1.12/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.12 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.12 / 12
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above Camarilla R1 + weekly uptrend + volume spike
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema50_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Camarilla S1 + weekly downtrend + volume spike
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema50_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below Camarilla S1 or weekly trend turns down
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above Camarilla R1 or weekly trend turns up
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals