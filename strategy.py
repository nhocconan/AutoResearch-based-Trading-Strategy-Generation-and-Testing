#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator + 1d KAMA trend filter with volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) for trend direction and entry timing,
# combined with 1d Kaufman Adaptive Moving Average for long-term trend filter.
# Volume filter ensures trades occur with sufficient market participation.
# Designed for low trade frequency (<30/year) to minimize fee drift.
# Williams Alligator is effective in trending markets while avoiding whipsaws in ranging conditions.
# KAMA adapts to market volatility, making it suitable for both bull and bear markets.

name = "12h_WilliamsAlligator_KAMA_Trend"
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
    
    # Williams Alligator components (13,8,5 periods with 8,5,3 shifts)
    close_series = pd.Series(close)
    jaw = close_series.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = close_series.rolling(window=8, min_periods=8).mean().shift(5).values
    lips = close_series.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # 1d Kaufman Adaptive Moving Average (KAMA)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # This needs correction
    # Proper ER calculation: |net change| / sum(|changes|) over period
    er = np.zeros_like(close_1d)
    for i in range(2, len(close_1d)):  # Start from index 2 for 3-period calculation
        net_change = np.abs(close_1d[i] - close_1d[i-2])
        sum_changes = np.abs(close_1d[i] - close_1d[i-1]) + np.abs(close_1d[i-1] - close_1d[i-2])
        if sum_changes > 0:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    # Align KAMA to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(kama_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Alligator signals: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        if position == 0:
            # LONG: Lips > Teeth > Jaw (uptrend) AND price > KAMA (long-term uptrend) with volume
            if lips[i] > teeth[i] > jaw[i] and close[i] > kama_1d_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (downtrend) AND price < KAMA (long-term downtrend) with volume
            elif lips[i] < teeth[i] < jaw[i] and close[i] < kama_1d_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Lips < Teeth (loss of uptrend momentum) OR price < KAMA
            if lips[i] < teeth[i] or close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Lips > Teeth (loss of downtrend momentum) OR price > KAMA
            if lips[i] > teeth[i] or close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals