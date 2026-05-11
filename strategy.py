#!/usr/bin/env python3
"""
6h_KAMA_Trend_Regime_Filter
Hypothesis: Use Kaufman's Adaptive Moving Average (KAMA) as primary trend filter on 6h timeframe.
Combine with Choppiness Index regime filter to distinguish trending vs ranging markets.
Only take long signals when KAMA slope is positive and market is trending (CHOP < 45).
Only take short signals when KAMA slope is negative and market is trending (CHOP < 45).
This avoids whipsaws in ranging markets while capturing strong trends in both bull and bear markets.
The strategy uses volume confirmation to ensure breakouts have institutional backing.
Target: 20-60 trades per year on 6h timeframe.
"""

name = "6h_KAMA_Trend_Regime_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA Calculation (10-period ER, 2/30 SC) ===
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # Will fix below
    
    # Recalculate volatility properly
    volatility = np.zeros(n)
    for i in range(10, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing Constants
    sc_fast = 2 / (2 + 1)   # EMA(2)
    sc_slow = 2 / (30 + 1)  # EMA(30)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
    
    # KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (trend direction)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # === Choppiness Index (14-period) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR (14-period)
    atr = np.zeros(n)
    atr[13] = np.mean(tr[1:14])  # Simple average for first value
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    sum_atr = np.zeros(n)
    for i in range(13, n):
        if i == 13:
            sum_atr[i] = np.sum(atr[1:14])
        else:
            sum_atr[i] = sum_atr[i-1] - atr[i-14] + atr[i]
    
    # Max/Min range over 14 periods
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(n):
        if i < 13:
            max_high[i] = np.max(high[:i+1])
            min_low[i] = np.min(low[:i+1])
        else:
            max_high[i] = np.max(high[i-13:i+1])
            min_low[i] = np.min(low[i-13:i+1])
    
    range_max_min = max_high - min_low
    chop = np.where(range_max_min != 0, 100 * np.log10(sum_atr / range_max_min) / np.log10(14), 50)
    
    # === Volume Filter (2.0x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_slope[i]) or np.isnan(chop[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: KAMA slope up + trending market (low chop) + volume
            if (kama_slope[i] > 0 and 
                chop[i] < 45 and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA slope down + trending market (low chop) + volume
            elif (kama_slope[i] < 0 and 
                  chop[i] < 45 and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA slope turns down OR market becomes ranging (high chop)
            if (kama_slope[i] <= 0) or (chop[i] >= 55):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: KAMA slope turns up OR market becomes ranging (high chop)
            if (kama_slope[i] >= 0) or (chop[i] >= 55):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals