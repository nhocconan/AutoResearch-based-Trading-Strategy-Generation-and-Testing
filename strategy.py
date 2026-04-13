#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout
Hypothesis: Uses weekly Camarilla pivot levels (H4, L4) as support/resistance on daily timeframe.
Breakout above H4 or below L4 with volume > 1.5x 20-day average triggers entry.
Trades in both bull and bear markets by capturing volatility expansion after pivot level breaks.
Target: 20-25 trades/year on daily (80-100 total over 4 years).
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
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels (H4, L4) from previous week
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    camarilla_h4 = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_l4 = close_1w - 1.1 * (high_1w - low_1w) / 2
    
    # Align weekly Camarilla levels to daily timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Volume confirmation: 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or np.isnan(volume_expansion[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above H4 with volume expansion
        if close[i] > camarilla_h4_aligned[i] and volume_expansion[i]:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        # Short entry: price breaks below L4 with volume expansion
        elif close[i] < camarilla_l4_aligned[i] and volume_expansion[i]:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        # Hold current position
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_1w_Camarilla_Pivot_Breakout"
timeframe = "1d"
leverage = 1.0