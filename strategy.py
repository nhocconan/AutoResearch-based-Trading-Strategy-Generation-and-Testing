#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_VolumeSpike_Bias_v1
Camarilla pivot breakout with volume spike and market bias filter.
Uses daily Camarilla levels (R1/S1) for entry, volume spike for confirmation,
and 1-week EMA trend filter to align with higher timeframe bias.
Designed for low-frequency, high-conviction trades in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === 1d Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = close_1d + (range_hl * 1.1 / 12)
    s1 = close_1d - (range_hl * 1.1 / 12)
    
    # === 1w EMA for trend bias (weekly close trend) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if i == 0:
            ema_1w[i] = close_1w[i]
        else:
            ema_1w[i] = ema_1w[i-1] + (2 / (50 + 1)) * (close_1w[i] - ema_1w[i-1])
    
    # === Volume spike (24-period average on 12h) ===
    vol_ma_24 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 24:
            vol_ma_24[i] = np.mean(volume[i-23:i+1])
        elif i > 0:
            vol_ma_24[i] = np.mean(volume[max(0, i-11):i+1])
        else:
            vol_ma_24[i] = volume[0]
    
    vol_confirm = volume > vol_ma_24 * 2.0  # volume spike: 2x average
    
    # === Align HTF data to 12h timeframe ===
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume spike AND weekly bias up
            if (close[i] > r1_aligned[i] and 
                vol_confirm[i] and 
                close[i] > ema_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume spike AND weekly bias down
            elif (close[i] < s1_aligned[i] and 
                  vol_confirm[i] and 
                  close[i] < ema_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns below R1 OR weekly bias flips down
            if (close[i] < r1_aligned[i] or 
                close[i] < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1 OR weekly bias flips up
            if (close[i] > s1_aligned[i] or 
                close[i] > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_VolumeSpike_Bias_v1"
timeframe = "12h"
leverage = 1.0