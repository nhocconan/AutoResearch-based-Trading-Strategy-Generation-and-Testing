#!/usr/bin/env python3
"""
4h_PivotReversal_WithTrendAndVolume
Hypothesis: Price reverses from daily pivot support/resistance levels with confirmation from 1d EMA trend and volume spike. 
Works in bull/bear by only taking trades in direction of daily trend (EMA50). 
Pivots provide institutional reference points; reversals with volume capture swing points in ranging markets.
Target: 20-35 trades/year (80-140 total) to minimize fee drag.
"""

name = "4h_PivotReversal_WithTrendAndVolume"
timeframe = "4h"
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
    
    # 1d data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot points (standard calculation)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d      # Resistance 1
    s1 = 2 * pivot - high_1d     # Support 1
    r2 = pivot + (high_1d - low_1d)  # Resistance 2
    s2 = pivot - (high_1d - low_1d)  # Support 2
    
    # 1d EMA50 for trend filter
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # 1d volume SMA20 for volume confirmation
    vol_sma20_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        vol_sma20_1d[19] = np.mean(df_1d['volume'].values[:20])
        for i in range(20, len(df_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + df_1d['volume'].values[i]) / 20
    
    # Align 1d indicators to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or \
           np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled)
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 6.0  # 6x 4h bars in 1d
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend and price relative to pivot levels
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        near_s1 = abs(close[i] - s1_aligned[i]) / close[i] < 0.005  # Within 0.5% of S1
        near_s2 = abs(close[i] - s2_aligned[i]) / close[i] < 0.005  # Within 0.5% of S2
        near_r1 = abs(close[i] - r1_aligned[i]) / close[i] < 0.005  # Within 0.5% of R1
        near_r2 = abs(close[i] - r2_aligned[i]) / close[i] < 0.005  # Within 0.5% of R2
        
        if position == 0:
            # Long: price near S1/S2 support, in uptrend, with volume
            if (near_s1 or near_s2) and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price near R1/R2 resistance, in downtrend, with volume
            elif (near_r1 or near_r2) and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price moves near pivot/resistance or trend turns down
            near_r1_exit = abs(close[i] - r1_aligned[i]) / close[i] < 0.005
            near_pivot_exit = abs(close[i] - pivot_aligned[i]) / close[i] < 0.005
            if near_r1_exit or near_pivot_exit or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price moves near pivot/support or trend turns up
            near_s1_exit = abs(close[i] - s1_aligned[i]) / close[i] < 0.005
            near_pivot_exit = abs(close[i] - pivot_aligned[i]) / close[i] < 0.005
            if near_s1_exit or near_pivot_exit or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals