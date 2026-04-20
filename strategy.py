#!/usr/bin/env python3
# 12h_1d_Camarilla_Pivot_R1_S1_Breakout_Volume_Confirmation
# Hypothesis: 12h breakout above Camarilla R1 or below S1 (from prior 1d) with volume confirmation.
# In bull/bear markets: long on R1 breakout with volume, short on S1 breakdown with volume.
# Camarilla levels provide statistically significant support/resistance; volume confirms breakout strength.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_1d_Camarilla_Pivot_R1_S1_Breakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 (using prior day's OHLC)
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA
    
    for i in range(start_idx, n):
        # Skip if Camarilla levels not available
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if close[i] > r1_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif close[i] < s1_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns below R1 or volume fails
            if close[i] < r1_aligned[i] or not volume_ok[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns above S1 or volume fails
            if close[i] > s1_aligned[i] or not volume_ok[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals