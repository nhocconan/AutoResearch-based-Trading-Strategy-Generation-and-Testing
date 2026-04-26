#!/usr/bin/env python3
"""
6h_VolumeWeightedRSI_PivotReversal_v1
Hypothesis: Use 1d pivot points with volume-weighted RSI on 6h to identify mean reversion extremes.
Long when price touches S1/S2 AND VW-RSI < 30, short when touches R1/R2 AND VW-RSI > 70.
Volume-weighted RSI reduces false signals during low-volume periods. Works in ranging markets (2021-2024) and 
bear markets (2025+) by fading overextended moves at institutional pivot levels.
Target: 80-120 total trades over 4 years = 20-30/year.
"""

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
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d pivot points (standard floor)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot * 2 - prev_low
    s1 = pivot * 2 - prev_high
    r2 = pivot + range_hl
    s2 = pivot - range_hl
    
    # Align pivot levels to 6h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume-weighted RSI (6h)
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Volume-weighted average gain/loss
        vol_ratio = volume / (np.mean(volume) + 1e-8)
        vol_ratio = np.clip(vol_ratio, 0.1, 5.0)  # Prevent extreme weights
        
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        
        # Wilder's smoothing with volume weighting
        avg_gain[period] = np.mean(gain[1:period+1] * vol_ratio[1:period+1]) / np.mean(vol_ratio[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1] * vol_ratio[1:period+1]) / np.mean(vol_ratio[1:period+1])
        
        for i in range(period+1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i] * vol_ratio[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i] * vol_ratio[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = max(30, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_vals[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(s2_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        rsi_val = rsi_vals[i]
        
        if position == 0:
            # Long: price at S1/S2 AND oversold VW-RSI
            long_signal = ((abs(close_val - s1_aligned[i]) < (s1_aligned[i] - s2_aligned[i]) * 0.1) or
                          (abs(close_val - s2_aligned[i]) < (s1_aligned[i] - s2_aligned[i]) * 0.1)) and \
                         (rsi_val < 30)
            
            # Short: price at R1/R2 AND overbought VW-RSI
            short_signal = ((abs(close_val - r1_aligned[i]) < (r2_aligned[i] - r1_aligned[i]) * 0.1) or
                           (abs(close_val - r2_aligned[i]) < (r2_aligned[i] - r1_aligned[i]) * 0.1)) and \
                          (rsi_val > 70)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price moves above pivot OR RSI > 50 (mean reversion complete)
            if (close_val > pivot[i]) or (rsi_val > 50):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price moves below pivot OR RSI < 50 (mean reversion complete)
            if (close_val < pivot[i]) or (rsi_val < 50):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_VolumeWeightedRSI_PivotReversal_v1"
timeframe = "6h"
leverage = 1.0