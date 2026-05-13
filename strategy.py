#!/usr/bin/env python3
# 1d_KAMA_1wTrend_Volume
# Hypothesis: Use KAMA direction from 1w trend filter on daily chart with volume confirmation. 
# Long when KAMA trending up and price above KAMA with volume spike. 
# Short when KAMA trending down and price below KAMA with volume spike.
# Exit when price crosses KAMA in opposite direction.
# Designed for low trade frequency (30-100 total trades over 4 years) to minimize fee drag in bear markets.

name = "1d_KAMA_1wTrend_Volume"
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

    # Get 1w data for KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1w close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(df_1w['close'], prepend=df_1w['close'][0]))
    volatility = np.abs(np.diff(df_1w['close'])).rolling(window=10, min_periods=10).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(df_1w['close'])
    kama[0] = df_1w['close'].iloc[0]
    for i in range(1, len(df_1w)):
        kama[i] = kama[i-1] + sc[i] * (df_1w['close'].iloc[i] - kama[i-1])
    
    # Align 1w KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)

    # Volume filter: >1.8x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA + KAMA trending up (today > yesterday) + volume spike
            if (close[i] > kama_aligned[i] and 
                kama_aligned[i] > kama_aligned[i-1] and
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + KAMA trending down (today < yesterday) + volume spike
            elif (close[i] < kama_aligned[i] and 
                  kama_aligned[i] < kama_aligned[i-1] and
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals