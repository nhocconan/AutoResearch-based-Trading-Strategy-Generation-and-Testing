#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Breakout_Volume_Confirmation
Hypothesis: Camarilla pivot levels on 12h with volume confirmation on 4h capture institutional breakouts.
Works in bull markets via breaks above H4 resistance and in bear markets via breaks below L4 support.
Uses 12h Camarilla levels, 4h volume > 1.5x 20-period average, and price close confirmation.
Target: 20-40 trades/year per symbol.
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
    
    # Calculate 12h Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        h4 = l4 = h3 = l3 = np.full(len(prices), np.nan)
    else:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # Calculate pivot and ranges
        pivot = (high_12h + low_12h + close_12h) / 3
        range_ = high_12h - low_12h
        
        # Camarilla levels
        h3 = pivot + (range_ * 1.1 / 6)
        l3 = pivot - (range_ * 1.1 / 6)
        h4 = pivot + (range_ * 1.1 / 2)
        l4 = pivot - (range_ * 1.1 / 2)
        
        # Align to 4h timeframe
        h3 = align_htf_to_ltf(prices, df_12h, h3)
        l3 = align_htf_to_ltf(prices, df_12h, l3)
        h4 = align_htf_to_ltf(prices, df_12h, h4)
        l4 = align_htf_to_ltf(prices, df_12h, l4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(h4[i]) or np.isnan(l4[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: break above H4 with volume expansion
        long_signal = (close[i] > h4[i] and volume_expansion[i])
        
        # Short signal: break below L4 with volume expansion
        short_signal = (close[i] < l4[i] and volume_expansion[i])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_12h_Camarilla_Pivot_Breakout_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0