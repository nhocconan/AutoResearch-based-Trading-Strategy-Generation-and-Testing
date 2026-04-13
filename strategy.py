#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout
Hypothesis: Uses 1-day Camarilla pivot levels for breakout trading on 4h timeframe.
In trending markets, price often tests and breaks through pivot levels (H4, L4).
Strong volume confirms breakout authenticity. Works in both bull and bear markets
by capturing momentum after consolidation around key support/resistance.
Target: 25-30 trades/year on 4h (100-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Classic formula: H4 = Close + 1.5 * (High - Low)
    #                L4 = Close - 1.5 * (High - Low)
    camarilla_high = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_low = close_1d - 1.5 * (high_1d - low_1d)
    
    # Get 4h data for breakout confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period volume average for confirmation
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean()
    volume_expansion_4h = volume_4h > (vol_ma_20_4h * 1.8)  # Strong volume threshold
    
    # Align all signals to 4h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    volume_expansion_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_expansion_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    # Track pivot levels for breakout detection
    camarilla_high_level = np.full(n, np.nan)
    camarilla_low_level = np.full(n, np.nan)
    
    for i in range(100, n):
        # Skip if data not ready
        if np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or np.isnan(volume_expansion_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update pivot levels (use previous day's levels)
        camarilla_high_level[i] = camarilla_high_aligned[i]
        camarilla_low_level[i] = camarilla_low_aligned[i]
        
        # Breakout conditions with volume confirmation
        breakout_up = (close[i] > camarilla_high_level[i]) and volume_expansion_4h_aligned[i]
        breakout_down = (close[i] < camarilla_low_level[i]) and volume_expansion_4h_aligned[i]
        
        # Entry logic
        if breakout_up and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_down and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Camarilla_Pivot_Breakout"
timeframe = "4h"
leverage = 1.0