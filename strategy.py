#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Elder Ray (Bull/Bear Power) + 6h Donchian(20) breakout.
Long when 1d Bull Power > 0 (buying pressure) and price breaks above 6h Donchian upper band.
Short when 1d Bear Power < 0 (selling pressure) and price breaks below 6h Donchian lower band.
Elder Ray measures daily bull/bear strength via EMA13, Donchian provides structure.
Designed to work in both bull (buy on strength + breakout) and bear (sell on weakness + breakdown) markets.
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
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d_s = pd.Series(close_1d)
    ema13_1d = close_1d_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d Elder Ray components
    bull_power_1d = high_1d - ema13_1d  # >0 = buying pressure
    bear_power_1d = low_1d - ema13_1d   # <0 = selling pressure
    
    # Get 6h data for Donchian(20) breakout
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate 6h Donchian(20) bands
    high_ma_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Elder Ray to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Align 6h Donchian bands to 6h timeframe (no additional delay needed)
    upper_6h_aligned = align_htf_to_ltf(prices, df_6h, high_ma_20)
    lower_6h_aligned = align_htf_to_ltf(prices, df_6h, low_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # need enough for Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(upper_6h_aligned[i]) or 
            np.isnan(lower_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 1d Bull Power > 0 (buying pressure) + price breaks above 6h Donchian upper
            if (bull_power_1d_aligned[i] > 0 and 
                close[i] > upper_6h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: 1d Bear Power < 0 (selling pressure) + price breaks below 6h Donchian lower
            elif (bear_power_1d_aligned[i] < 0 and 
                  close[i] < lower_6h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 6h Donchian lower (mean reversion)
            if close[i] < lower_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 6h Donchian upper (mean reversion)
            if close[i] > upper_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dElderRay_Donchian20_Breakout"
timeframe = "6h"
leverage = 1.0