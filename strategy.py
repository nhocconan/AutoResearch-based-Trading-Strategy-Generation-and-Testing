#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot levels (R1/S1) from daily data with volume confirmation.
- Camarilla levels (R1/S1) capture institutional support/resistance from prior day's range
- Volume spike confirms breakout strength (>1.5x 20-period average volume)
- Trade in direction of breakout: long at R1 break, short at S1 break
- Exit on opposite level touch (S1 for longs, R1 for shorts) or when volume dries up
- Target: 15-25 trades/year to minimize fee drag on 12h timeframe
- Uses discrete position sizing (0.25) to reduce churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1) from previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        camarilla_r1[i] = prev_close + 1.1 * (prev_high - prev_low) / 12
        camarilla_s1[i] = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need at least 2 days of data
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 + volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price touches S1 (opposite level) or volume dries up
            if (close[i] <= camarilla_s1_aligned[i] or 
                not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches R1 (opposite level) or volume dries up
            if (close[i] >= camarilla_r1_aligned[i] or 
                not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0