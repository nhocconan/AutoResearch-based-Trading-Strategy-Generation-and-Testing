#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume
Hypothesis: Trade Camarilla pivot breakouts on 12h with volume confirmation. Camarilla levels (R1, S1) derived from 1d OHLC act as key support/resistance in both bull and bear markets. Enter long on break above R1 with volume > 2x 12-period average; short on break below S1 with volume > 2x average. Exit on opposite level touch (S1 for long, R1 for short). Uses 1d data for pivots, avoiding look-ahead via mtf_data alignment. Targets 15-30 trades/year via strict breakout conditions, reducing whipsaw and fee impact. Works in bull (breakouts continue) and bear (fades at S1/R1).
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 1d bar: R1, S1
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h = high_1d - low_1d
    r1_1d = close_1d + camarilla_h * 1.1 / 12
    s1_1d = close_1d - camarilla_h * 1.1 / 12
    
    # Align 1d Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: volume > 2x 12-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 12
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1)  # Need volume MA and aligned data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume
            if close[i] > r1_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume
            elif close[i] < s1_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches or crosses below S1
            if close[i] < s1_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches or crosses above R1
            if close[i] > r1_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0