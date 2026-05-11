#!/usr/bin/env python3
"""
4h_KAMA_Direction_1dTrend_VolumeSpike
Hypothesis: Trade in direction of Kaufman Adaptive Moving Average (KAMA) with 1d trend filter and volume spike confirmation. 
KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends. 
Combined with daily trend and volume confirmation, this should work in both bull and bear markets by filtering counter-trend moves.
Target: 20-30 trades/year on 4h to minimize fee drag.
"""

name = "4h_KAMA_Direction_1dTrend_VolumeSpike"
timeframe = "4h"
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
    
    # === Daily Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === KAMA (Kaufman Adaptive Moving Average) on 4h close ===
    # ER (Efficiency Ratio) = |change| / sum(|changes|) over fast period
    # Smoothing constant = [ER * (fastest SC - slowest SC) + slowest SC]^2
    # KAMA = previous KAMA + SC * (price - previous KAMA)
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    lookback = 10  # ER lookback period
    
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = change
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(lookback, n):
        net_change = abs(close[i] - close[i-lookback])
        total_change = np.sum(abs_change[i-lookback+1:i+1])
        if total_change > 0:
            er[i] = net_change / total_change
        else:
            er[i] = 0
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Volume Filter (2.0x 20-period EMA on 4h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers KAMA and daily calculations)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_4h[i]) or np.isnan(kama[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA with uptrend and volume spike
            if (close[i] > kama[i] and 
                close[i] > ema34_4h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with downtrend and volume spike
            elif (close[i] < kama[i] and 
                  close[i] < ema34_4h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals