#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_reversion_v1
# Strategy: 4h mean reversion at daily Camarilla pivot levels with volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Price tends to revert to the mean after reaching extreme Camarilla levels (H3/L3).
# In ranging markets, these levels act as support/resistance. Volume confirms institutional
# interest at these levels. Works in both bull/bear markets as it's a mean-reversion strategy.
# Designed for low trade frequency (<25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.0 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.0 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or \
           np.isnan(volume_ma[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * volume_ma[i]
        
        # Price near Camarilla levels (within 0.2% tolerance)
        near_h3 = abs(close[i] - camarilla_h3_aligned[i]) / camarilla_h3_aligned[i] < 0.002
        near_l3 = abs(close[i] - camarilla_l3_aligned[i]) / camarilla_l3_aligned[i] < 0.002
        
        # Entry conditions
        # Long: Price near L3 support with volume confirmation
        if near_l3 and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price near H3 resistance with volume confirmation
        elif near_h3 and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price moves back toward midpoint (mean reversion)
        elif position == 1 and close[i] > (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] < (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals