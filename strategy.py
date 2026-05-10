#!/usr/bin/env python3
"""
6h_MultiTimeframe_ElderRay_Strategy
Hypothesis: Elder Ray (Bull Power/Bear Power) from 1d data combined with 6s EMA crossover and volume confirmation.
Bull Power = High - EMA13, Bear Power = EMA13 - Low. 
Long when Bull Power > 0, EMA9 > EMA21, and volume > 1.5x average.
Short when Bear Power > 0, EMA9 < EMA21, and volume > 1.5x average.
Uses 1d Elder Ray for regime and 6s EMA for timing to work in both bull and bear markets.
Target: 20-30 trades/year (80-120 total) to minimize fee drag.
"""

name = "6h_MultiTimeframe_ElderRay_Strategy"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    volume_1d = df_1d['volume'].values
    
    # 1d EMA13 for Elder Ray
    ema13_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 13:
        ema13_1d[12] = np.mean(close_1d[:13])
        alpha = 2 / (13 + 1)
        for i in range(13, len(close_1d)):
            ema13_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema13_1d[i-1]
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_1d - ema13_1d
    bear_power = ema13_1d - low_1d
    
    # 1d volume SMA20 for volume confirmation
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    # 6s EMA9 and EMA21 for entry timing
    ema9 = np.full(n, np.nan)
    ema21 = np.full(n, np.nan)
    if n >= 21:
        ema9[8] = np.mean(close[:9])
        ema21[20] = np.mean(close[:21])
        alpha9 = 2 / (9 + 1)
        alpha21 = 2 / (21 + 1)
        for i in range(9, n):
            ema9[i] = alpha9 * close[i] + (1 - alpha9) * ema9[i-1]
        for i in range(21, n):
            ema21[i] = alpha21 * close[i] + (1 - alpha21) * ema21[i-1]
    
    # Align 1d indicators to 6s
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Wait for EMA21
    
    for i in range(start_idx, n):
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(ema9[i]) or np.isnan(ema21[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6s volume > 1.5x average 1d volume (scaled)
        # 4x 6s bars in 1d (24h / 6h = 4)
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 4.0
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # 6s EMA trend
        ema9_above_ema21 = ema9[i] > ema21[i]
        ema9_below_ema21 = ema9[i] < ema21[i]
        
        if position == 0:
            # Long: Bull Power > 0, EMA9 > EMA21, volume confirmation
            if bull_power_aligned[i] > 0 and ema9_above_ema21 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0, EMA9 < EMA21, volume confirmation
            elif bear_power_aligned[i] > 0 and ema9_below_ema21 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power <= 0 or EMA9 <= EMA21
            if bull_power_aligned[i] <= 0 or not ema9_above_ema21:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power <= 0 or EMA9 >= EMA21
            if bear_power_aligned[i] <= 0 or not ema9_below_ema21:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals