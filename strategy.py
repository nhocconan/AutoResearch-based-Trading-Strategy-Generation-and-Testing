#!/usr/bin/env python3
# 12h_KAMA_1dTrend_Volume
# Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) for trend direction with 1d trend filter and volume confirmation.
# Long when KAMA turns up (bullish) with price above 1d EMA50 and volume spike; short when KAMA turns down (bearish) with price below 1d EMA50 and volume spike.
# Exit when KAMA reverses direction or trend filter fails.
# KAMA adapts to market noise, reducing false signals in choppy conditions while capturing trends.
# Designed for low trade frequency (50-150 total trades over 4 years) with clear entry/exit rules to avoid overtrading.

name = "12h_KAMA_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate KAMA (Kaufman Adaptive Moving Average) on 12h close
    # KAMA parameters: ER (efficiency ratio) period=10, fast SC=2/(2+1), slow SC=2/(30+1)
    er_period = 10
    fast_sc = 2 / (2 + 1)  # 0.6667
    slow_sc = 2 / (30 + 1)  # 0.0645
    
    # Calculate change and volatility
    change = np.abs(np.diff(close, k=er_period))  # absolute change over er_period
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # sum of absolute changes over er_period
    
    # Handle the array operations properly
    change_padded = np.concatenate([np.full(er_period, np.nan), change])
    volatility_padded = np.concatenate([np.full(er_period, np.nan), 
                                       [np.sum(np.abs(np.diff(close[i-er_period:i+1]))) 
                                        for i in range(er_period-1, len(close))]])
    
    # Calculate efficiency ratio (ER)
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    
    # Calculate smoothing constant (SC)
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_period] = close[er_period]  # start with first close value
    
    for i in range(er_period + 1, len(close)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(er_period + 1, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA turning up (bullish) + price above 1d EMA50 (uptrend) + volume spike
            if (kama[i] > kama[i-1] and 
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turning down (bearish) + price below 1d EMA50 (downtrend) + volume spike
            elif (kama[i] < kama[i-1] and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down or trend fails (price below EMA50)
            if (kama[i] < kama[i-1] or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or trend fails (price above EMA50)
            if (kama[i] > kama[i-1] or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals