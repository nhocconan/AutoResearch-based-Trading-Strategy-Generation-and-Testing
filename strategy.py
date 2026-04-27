#!/usr/bin/env python3
"""
12h_4h_SqueezeBreakout_SqueezeExit
Hypothesis: Combine Bollinger Band squeeze (low volatility) with 4h Donchian breakout for directional entry, 
exit on Bollinger Band expansion (volatility increase). Uses 12h for Bollinger Band squeeze detection 
and 4h for Donchian breakout direction. Designed to capture explosive moves after low volatility periods 
in both bull and bear markets. Target: 20-40 trades/year.
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
    
    # Get 12h data for Bollinger Band squeeze detection
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Bollinger Bands (20, 2) on 12h
    bb_length = 20
    bb_mult = 2.0
    
    # Basis (SMA)
    basis_12h = np.full(len(close_12h), np.nan)
    for i in range(bb_length - 1, len(close_12h)):
        basis_12h[i] = np.mean(close_12h[i - bb_length + 1:i + 1])
    
    # Standard deviation
    dev_12h = np.full(len(close_12h), np.nan)
    for i in range(bb_length - 1, len(close_12h)):
        dev_12h[i] = np.std(close_12h[i - bb_length + 1:i + 1])
    
    # Upper and lower bands
    upper_12h = basis_12h + bb_mult * dev_12h
    lower_12h = basis_12h - bb_mult * dev_12h
    bb_width_12h = (upper_12h - lower_12h) / basis_12h  # Normalized width
    
    # Bollinger Band squeeze: low volatility condition
    # Squeeze when BB width is below 20-period average
    bb_width_ma_20 = np.full(len(bb_width_12h), np.nan)
    for i in range(19, len(bb_width_12h)):
        bb_width_ma_20[i] = np.mean(bb_width_12h[i - 19:i + 1])
    
    squeeze_condition = bb_width_12h < bb_width_ma_20
    
    # Get 4h data for Donchian breakout direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian Channel (20) on 4h
    donchian_length = 20
    upper_4h = np.full(len(high_4h), np.nan)
    lower_4h = np.full(len(low_4h), np.nan)
    
    for i in range(donchian_length - 1, len(high_4h)):
        upper_4h[i] = np.max(high_4h[i - donchian_length + 1:i + 1])
        lower_4h[i] = np.min(low_4h[i - donchian_length + 1:i + 1])
    
    # Align indicators to lower timeframe (assuming 1h primary for signal generation)
    # But we'll use 4h as primary timeframe according to instructions
    squeeze_aligned = align_htf_to_ltf(prices, df_12h, squeeze_condition.astype(float))
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup period
    start_idx = max(bb_length + 19, donchian_length + 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(squeeze_aligned[i]) or 
            np.isnan(upper_4h_aligned[i]) or 
            np.isnan(lower_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: squeeze condition + price breaks above upper Donchian
            if squeeze_aligned[i] > 0.5 and price > upper_4h_aligned[i]:
                signals[i] = size
                position = 1
            # Enter short: squeeze condition + price breaks below lower Donchian
            elif squeeze_aligned[i] > 0.5 and price < lower_4h_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: squeeze condition ends (volatility expansion) OR price breaks below lower Donchian
            if squeeze_aligned[i] <= 0.5 or price < lower_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: squeeze condition ends OR price breaks above upper Donchian
            if squeeze_aligned[i] <= 0.5 or price > upper_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_4h_SqueezeBreakout_SqueezeExit"
timeframe = "4h"
leverage = 1.0