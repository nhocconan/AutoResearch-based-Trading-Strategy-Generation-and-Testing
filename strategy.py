#!/usr/bin/env python3
"""
12h_200EMA_Cross_Volume_Spike
Hypothesis: 12h price crossing above/below 200 EMA with volume spike captures trend changes.
Works in bull/bear by trading only in direction of 200 EMA slope. Volume filter ensures momentum.
Target: 15-30 trades/year (60-120 total) to minimize fee drift.
"""

name = "12h_200EMA_Cross_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for 200 EMA and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 200 EMA on daily close
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema200_1d[i-1]
    
    # 20-day volume SMA on daily volume
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    # Align to 12h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 12h volume > 2x average daily volume (scaled to 12h)
        # 12h is 0.5 day, so scale daily volume to 12h equivalent
        vol_12h_scaled = vol_sma20_1d_aligned[i] * 0.5
        volume_spike = volume[i] > 2.0 * vol_12h_scaled
        
        # Price relative to 200 EMA
        price_above_ema = close[i] > ema200_1d_aligned[i]
        price_below_ema = close[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # Long: price crosses above 200 EMA with volume spike
            if price_above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below 200 EMA with volume spike
            elif price_below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses back below 200 EMA
            if not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses back above 200 EMA
            if not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals