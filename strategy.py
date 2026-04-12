#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_volume
Hypothesis: 12-hour strategy using Camarilla pivot levels from daily timeframe for entry/exit, with volume confirmation.
Buys when price breaks above Camarilla H3 with volume surge, sells when breaks below L3.
Uses daily volatility filter to avoid choppy markets. Designed to work in both bull and bear markets by trading breakouts.
Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag.
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
    
    # Get daily data for Camarilla pivots and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels for previous day
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # H2 = close + 0.5 * (high - low)
    # H1 = close + 0.25 * (high - low)
    # L1 = close - 0.25 * (high - low)
    # L2 = close - 0.5 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    range_1d = high_1d - low_1d
    H3_1d = close_1d + 1.0 * range_1d
    L3_1d = close_1d - 1.0 * range_1d
    H4_1d = close_1d + 1.5 * range_1d
    L4_1d = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 12h timeframe
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    H4_1d_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    L4_1d_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    
    # Daily ATR for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(np.roll(high_1d, 1) - close_1d)
    tr3 = np.abs(np.roll(low_1d, 1) - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        # Skip if data not ready
        if (np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or 
            np.isnan(H4_1d_aligned[i]) or np.isnan(L4_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above 50% of its 50-period median
        atr_median = np.nanmedian(atr_1d_aligned[max(0, i-50):i+1])
        if atr_1d_aligned[i] < atr_median * 0.5:
            # Low volatility - hold position or stay flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x MA volume
        volume_ok = volume[i] > vol_ma[i] * 1.5
        
        # Breakout conditions
        if volume_ok:
            # Long breakout above H3
            if close[i] > H3_1d_aligned[i] and position != 1:
                position = 1
                signals[i] = 0.25
            # Short breakdown below L3
            elif close[i] < L3_1d_aligned[i] and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit on opposite breakout
            elif position == 1 and close[i] < L3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > H3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # No volume confirmation - hold or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_breakout_volume"
timeframe = "12h"
leverage = 1.0