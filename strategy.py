#!/usr/bin/env python3
"""
12h_1w_Camarilla_R1S1_Breakout_VolumeFilter - Weekly Camarilla pivot breakout with volume confirmation
Hypothesis: Weekly Camarilla R1/S1 levels act as strong support/resistance. Breaking these levels with
volume confirmation indicates institutional interest. Works in bull (breakouts continue) and bear
(failures at resistance/support) markets. 12h timeframe balances signal quality and trade frequency.
Target: 15-35 trades/year (60-140 total over 4 years).
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
    
    # === Weekly Camarilla Pivot Levels ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each weekly bar
    R1 = np.full_like(close_1w, np.nan)
    S1 = np.full_like(close_1w, np.nan)
    PP = np.full_like(close_1w, np.nan)
    
    for i in range(len(close_1w)):
        if i >= 0:  # Need at least one bar
            typical_price = (high_1w[i] + low_1w[i] + close_1w[i]) / 3
            range_ = high_1w[i] - low_1w[i]
            PP[i] = typical_price
            R1[i] = PP[i] + (range_ * 1.1 / 12)
            S1[i] = PP[i] - (range_ * 1.1 / 12)
    
    # Align weekly levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    PP_aligned = align_htf_to_ltf(prices, df_1w, PP)
    
    # === 12h Volume Confirmation ===
    # 24-period average volume (2 periods of 12h = 1 day)
    vol_ma_24 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 23:
            vol_ma_24[i] = np.mean(volume[i-23:i+1])
        elif i > 0:
            vol_ma_24[i] = np.mean(volume[max(0, i-11):i+1])
        else:
            vol_ma_24[i] = volume[0]
    
    # Volume spike: current volume > 2.0x average
    volume_spike = volume > vol_ma_24 * 2.0
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Close breaks above R1 with volume spike
            if close[i] > R1_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Close breaks below S1 with volume spike
            elif close[i] < S1_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Close crosses below weekly pivot point
            if close[i] < PP_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close crosses above weekly pivot point
            if close[i] > PP_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Camarilla_R1S1_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0