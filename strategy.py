#!/usr/bin/env python3
"""
12h_KAMA_Direction_With_Trend_and_Volume
Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies trend direction with low whipsaw.
Trend filter from 1d EMA34 ensures alignment with higher timeframe momentum.
Volume spike confirms institutional participation.
Long when KAMA rising, price > KAMA, volume > 1.5x average, and 1d uptrend.
Short when KAMA falling, price < KAMA, volume > 1.5x average, and 1d downtrend.
Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets via trend filter and volume confirmation.
"""

name = "12h_KAMA_Direction_With_Trend_and_Volume"
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
    
    # Kaufman Adaptive Moving Average (KAMA)
    def kama(close, er_period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, er_period=10, fast=2, slow=30)
    kama_rising = kama_vals > np.roll(kama_vals, 1)
    kama_falling = kama_vals < np.roll(kama_vals, 1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_avg = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_spike = volume > (1.5 * vol_avg)
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = df_1d['close'].values > ema_34_1d
    downtrend_1d = df_1d['close'].values < ema_34_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if position == 0:
            # LONG: KAMA rising, price > KAMA, volume spike, 1d uptrend
            if kama_rising[i] and close[i] > kama_vals[i] and vol_spike[i] and uptrend_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, price < KAMA, volume spike, 1d downtrend
            elif kama_falling[i] and close[i] < kama_vals[i] and vol_spike[i] and downtrend_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling or price < KAMA
            if kama_falling[i] or close[i] < kama_vals[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising or price > KAMA
            if kama_rising[i] or close[i] > kama_vals[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals