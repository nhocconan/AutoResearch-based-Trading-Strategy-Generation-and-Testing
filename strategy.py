#!/usr/bin/env python3
"""
6h_Liquidity_Grab_Reversal
Hypothesis: Price often sweeps liquidity (equal highs/lows) before reversing.
Look for equal highs/lows formation on 1d, then wait for 6h price to breach that level
with volume spike, then enter reversal. Works in ranging markets (common in 2025+)
and during pullbacks in trends. Targets 50-120 trades over 4 years (12-30/year).
"""

name = "6h_Liquidity_Grab_Reversal"
timeframe = "6h"
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
    
    # 1d equal highs/lows detection (liquidity zones)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Find equal highs/lows within 0.1% tolerance
    equal_high = np.zeros(len(high_1d), dtype=bool)
    equal_low = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d)):
        # Equal high: current high within 0.1% of previous high
        if abs(high_1d[i] - high_1d[i-1]) / high_1d[i-1] < 0.001:
            equal_high[i] = True
        # Equal low: current low within 0.1% of previous low
        if abs(low_1d[i] - low_1d[i-1]) / low_1d[i-1] < 0.001:
            equal_low[i] = True
    
    # Store liquidity levels
    liq_high = np.where(equal_high, high_1d, np.nan)
    liq_low = np.where(equal_low, low_1d, np.nan)
    
    # Forward fill to maintain levels until broken
    for i in range(1, len(liq_high)):
        if not np.isnan(liq_high[i-1]) and np.isnan(liq_high[i]):
            liq_high[i] = liq_high[i-1]
        if not np.isnan(liq_low[i-1]) and np.isnan(liq_low[i]):
            liq_low[i] = liq_low[i-1]
    
    # Align to 6h
    liq_high_aligned = align_htf_to_ltf(prices, df_1d, liq_high)
    liq_low_aligned = align_htf_to_ltf(prices, df_1d, liq_low)
    
    # 1d volume spike detection
    vol_ma20 = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_ma20[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_ma20[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 2)  # Need volume MA and lookback
    
    for i in range(start_idx, n):
        if np.isnan(liq_high_aligned[i]) or np.isnan(liq_low_aligned[i]) or \
           np.isnan(vol_ma20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 6h volume > 2x average 1d volume (scaled)
        vol_6h_approx = vol_ma20_aligned[i] / 4.0  # 4x 6h in 1d
        volume_spike = volume[i] > 2.0 * vol_6h_approx
        
        if position == 0:
            # Long: price takes out liquidity low then reverses up
            if (low[i] <= liq_low_aligned[i] * 1.001 and  # breached low
                close[i] > liq_low_aligned[i] and        # closed back above
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price takes out liquidity high then reverses down
            elif (high[i] >= liq_high_aligned[i] * 0.999 and  # breached high
                  close[i] < liq_high_aligned[i] and          # closed back below
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price takes out liquidity high or loses momentum
            if high[i] >= liq_high_aligned[i] * 0.999 or volume[i] < vol_6h_approx:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price takes out liquidity low or loses momentum
            if low[i] <= liq_low_aligned[i] * 1.001 or volume[i] < vol_6h_approx:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals