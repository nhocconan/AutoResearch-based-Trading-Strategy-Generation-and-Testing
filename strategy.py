#!/usr/bin/env python3
"""
4h_50_150_MA_Crossover_Volume_Trend
Hypothesis: Use 50-period and 150-period EMA crossovers on 4h timeframe with 1d trend filter and volume confirmation to capture medium-term trends.
The 50/150 EMA crossover is a proven trend-following signal that works in both bull and bear markets when filtered by higher timeframe trend and volume.
Target: 20-35 trades/year (80-140 total) to minimize fee drag while maintaining profitability.
"""

name = "4h_50_150_MA_Crossover_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 50 EMA on close
    ema50 = np.full(n, np.nan)
    if n >= 50:
        ema50[49] = np.mean(close[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, n):
            ema50[i] = alpha * close[i] + (1 - alpha) * ema50[i-1]
    
    # 150 EMA on close
    ema150 = np.full(n, np.nan)
    if n >= 150:
        ema150[149] = np.mean(close[:150])
        alpha = 2 / (150 + 1)
        for i in range(150, n):
            ema150[i] = alpha * close[i] + (1 - alpha) * ema150[i-1]
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
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
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 150  # Wait for EMA150
    
    for i in range(start_idx, n):
        if np.isnan(ema50[i]) or np.isnan(ema150[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled)
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 6.0  # 6x 4h bars in 1d
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend filter: price above/below 1d EMA50
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        # EMA crossover signals
        golden_cross = ema50[i] > ema150[i] and ema50[i-1] <= ema150[i-1]
        death_cross = ema50[i] < ema150[i] and ema50[i-1] >= ema150[i-1]
        
        if position == 0:
            # Long: golden cross, in uptrend, with volume
            if golden_cross and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: death cross, in downtrend, with volume
            elif death_cross and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: death cross or trend turns down
            if death_cross or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: golden cross or trend turns up
            if golden_cross or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals