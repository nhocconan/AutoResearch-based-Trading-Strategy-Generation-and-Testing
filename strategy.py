#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Filter_v3
Hypothesis: Trade breakouts of Camarilla R1/S1 levels derived from 1d OHLC, confirmed by volume > 2x 24-period average. Only trade long when price breaks above R1, short when breaks below S1. This strategy targets 20-40 trades/year by requiring both price level breach and strong volume confirmation, reducing false breakouts. Works in bull/bear markets by following institutional price levels that act as support/resistance regardless of trend. Uses 1d timeframe for pivot calculation to avoid noise.
"""

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 1d bar: R1, S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            camarilla_r1[i] = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 12
            camarilla_s1[i] = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 2x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1)  # Need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume
            if close[i] > camarilla_r1_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume
            elif close[i] < camarilla_s1_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 (reversal signal)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (reversal signal)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Filter_v3"
timeframe = "4h"
leverage = 1.0