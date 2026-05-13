#!/usr/bin/env python3
# 12h_KAMA_Trend_Volume_1dTrend_Filter
# Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) on 12h to capture trend with low lag in whipsaws,
# combined with 1d EMA trend filter and volume confirmation. KAMA adapts to market noise, reducing false
# breakouts during chop while maintaining sensitivity to strong trends. Works in bull (follows KAMA up with
# bullish 1d trend) and bear (avoids false signals via 1d filter). Target: 50-150 total trades over 4 years.

name = "12h_KAMA_Trend_Volume_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_len=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    er = np.zeros_like(close)
    for i in range(er_len, len(close)):
        if volatility[i-er_len] != 0:
            er[i] = change[i] / volatility[i-er_len]
        else:
            er[i] = 0
    
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Calculate KAMA on 12h data
    kama = calculate_kama(close, er_len=10, fast_sc=2, slow_sc=30)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after KAMA warmup
        # Skip if any required value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(kama[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price above KAMA + price above 1d EMA (bullish trend) + volume spike
            if (close[i] > kama[i] and 
                close[i] > ema_34_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price below KAMA + price below 1d EMA (bearish trend) + volume spike
            elif (close[i] < kama[i] and 
                  close[i] < ema_34_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price below KAMA or price below 1d EMA
            if (close[i] < kama[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price above KAMA or price above 1d EMA
            if (close[i] > kama[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals