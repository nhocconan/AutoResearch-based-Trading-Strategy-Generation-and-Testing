#!/usr/bin/env python3
# 6h_Liquidity_Sweep_Volume_Confirmation
# Hypothesis: 6-hour liquidity sweeps (stop hunts) followed by reversal with volume confirmation.
# Liquidity sweep identified when price breaks recent swing high/low but closes back within range,
# indicating stop hunt. Entry taken in opposite direction of sweep with volume confirmation.
# Works in both bull/bear markets as it exploits market structure breaks and reversals.

name = "6h_Liquidity_Sweep_Volume_Confirmation"
timeframe = "6h"
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
    
    # Calculate 20-period swing high/low for liquidity detection
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    swing_high = rolling_max(high, 20)
    swing_low = rolling_min(low, 20)
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough history for swing calculations
    
    for i in range(start_idx, n):
        if np.isnan(swing_high[i]) or np.isnan(swing_low[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish liquidity sweep: price breaks above swing high but closes below it (stop hunt)
            if high[i] > swing_high[i] and close[i] < swing_high[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Bearish liquidity sweep: price breaks below swing low but closes above it (stop hunt)
            elif low[i] < swing_low[i] and close[i] > swing_low[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below swing low or reverse sweep occurs
            if low[i] < swing_low[i] and close[i] > swing_low[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above swing high or reverse sweep occurs
            if high[i] > swing_high[i] and close[i] < swing_high[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals