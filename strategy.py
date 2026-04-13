#!/usr/bin/env python3
"""
12h_1d_Donchian_Breakout_Volume
Hypothesis: Donchian(20) breakouts on 12h timeframe capture medium-term trends.
Volume confirmation filters false breakouts, while 1d ATR filter ensures trades occur
only during sufficient volatility. Works in bull markets via trend continuation and
in bear markets via breakdowns, with volatility filter avoiding ranging markets.
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
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian(20) on 12h: upper/lower bands
    # Use rolling window with min_periods=20
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 12h timeframe (already aligned by get_htf_data)
    upper_20 = high_20
    lower_20 = low_20
    
    # Get 1d data for ATR(14) volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current 12h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is not ready
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Skip if volatility is too low (ATR below 50-period MA)
        if i >= 50:
            atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
            if np.isnan(atr_ma_50[i]) or atr_14[i] < atr_ma_50[i]:
                signals[i] = 0.0
                continue
        
        # Long conditions:
        # 1. Breakout above 12h Donchian upper band with volume expansion
        # 2. Close must be above the breakout level (avoid fakeouts)
        breakout_long = (close[i] > upper_20[i]) and volume_expansion[i]
        
        # Short conditions:
        # 1. Breakdown below 12h Donchian lower band with volume expansion
        # 2. Close must be below the breakdown level (avoid fakeouts)
        breakdown_short = (close[i] < lower_20[i]) and volume_expansion[i]
        
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakdown_short and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_Donchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0