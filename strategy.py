#!/usr/bin/env python3
"""
12h_1d_Pivot_R1_S1_Breakout_Volume
Hypothesis: Use daily Camarilla pivot levels (R1/S1) as support/resistance. 
Go long when price breaks above daily R1 with volume > 1.5x 20-period average.
Go short when price breaks below daily S1 with volume > 1.5x 20-period average.
Exit on opposite pivot touch or volume divergence.
Works in bull markets via breakouts and in bear via mean reversion at S1/R1.
Target: 12-37 trades/year by requiring pivot break + volume confirmation.
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) for each day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    
    valid = ~(np.isnan(high_1d) | np.isnan(low_1d) | np.isnan(close_1d))
    camarilla_r1[valid] = close_1d[valid] + 1.1 * (high_1d[valid] - low_1d[valid]) / 12
    camarilla_s1[valid] = close_1d[valid] - 1.1 * (high_1d[valid] - low_1d[valid]) / 12
    
    # Align daily pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1)  # Need volume MA and at least one pivot
    
    for i in range(start_idx, n):
        # Skip if pivot data not available
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: break above R1 with volume
            if close[i] > r1_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume
            elif close[i] < s1_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches or breaks below S1
            if close[i] < s1_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches or breaks above R1
            if close[i] > r1_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Pivot_R1_S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0