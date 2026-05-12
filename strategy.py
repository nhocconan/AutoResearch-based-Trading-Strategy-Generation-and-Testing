#!/usr/bin/env python3
"""
12h_KAMA_Direction_Plus_Volume_Confirmation
Hypothesis: On 12h timeframe, use KAMA (Kaufman Adaptive Moving Average) to determine trend direction on 1d timeframe. Enter long when price crosses above KAMA with volume > 1.5x average, short when price crosses below KAMA with volume > 1.5x average. Exit when price crosses back below/above KAMA. Uses adaptive trend filter to reduce whipsaw in sideways markets while capturing strong trends. Targets 15-25 trades per year to minimize fee drag and improve robustness across bull/bear markets.
"""

name = "12h_KAMA_Direction_Plus_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_ema=2, slow_ema=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    
    # Calculate Efficiency Ratio
    er = np.zeros_like(close)
    for i in range(er_period, len(close)):
        if np.sum(volatility[i-er_period+1:i+1]) > 0:
            er[i] = np.abs(close[i] - close[i-er_period]) / np.sum(volatility[i-er_period+1:i+1])
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate KAMA on 1d close
    kama_1d = calculate_kama(close_1d, er_period=10, fast_ema=2, slow_ema=30)
    
    # Align KAMA to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above KAMA + volume confirmation
            if (close[i] > kama_1d_aligned[i] and 
                close[i-1] <= kama_1d_aligned[i-1] and 
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA + volume confirmation
            elif (close[i] < kama_1d_aligned[i] and 
                  close[i-1] >= kama_1d_aligned[i-1] and 
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama_1d_aligned[i] and close[i-1] >= kama_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama_1d_aligned[i] and close[i-1] <= kama_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals