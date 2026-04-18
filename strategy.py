#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Momentum_v1
Strategy: 12h Camarilla pivot breakout with momentum confirmation.
Long: Price breaks above R1 (bullish pivot) with momentum confirmation.
Short: Price breaks below S1 (bearish pivot) with momentum confirmation.
Uses 1d pivot levels, 12h momentum (ROC > 0), and volume filter.
Designed for 12h timeframe: ~10-20 trades/year per symbol (40-80 total over 4 years).
Works in bull/bear via momentum confirmation and pivot-based mean reversion.
"""

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Momentum confirmation: 12-period ROC on 12h timeframe
    roc = np.zeros_like(close)
    roc[12:] = (close[12:] - close[:-12]) / close[:-12] * 100
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    vol_ma[20:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma = np.concatenate([np.full(19, np.nan), vol_ma])
    vol_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 12)  # need volume MA and ROC
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(roc[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Momentum confirmation: positive ROC for long, negative for short
        mom_long = roc[i] > 0
        mom_short = roc[i] < 0
        
        # Breakout conditions
        breakout_long = close[i] > r1_aligned[i]
        breakout_short = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume + momentum
            if breakout_long and vol_confirm[i] and mom_long:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume + momentum
            elif breakout_short and vol_confirm[i] and mom_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below S1 or momentum reversal
            if close[i] < s1_aligned[i] or roc[i] < 0:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above R1 or momentum reversal
            if close[i] > r1_aligned[i] or roc[i] > 0:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_Momentum_v1"
timeframe = "12h"
leverage = 1.0