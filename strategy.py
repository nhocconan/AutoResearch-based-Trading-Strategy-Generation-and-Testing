#!/usr/bin/env python3
name = "6h_Williams_Alligator_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h and 1d data
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Williams Alligator from 12h: Jaw (13), Teeth (8), Lips (5)
    close_12h = df_12h['close'].values
    jaw = pd.Series(close_12h).rolling(window=13, center=False).mean().shift(8).values
    teeth = pd.Series(close_12h).rolling(window=8, center=False).mean().shift(5).values
    lips = pd.Series(close_12h).rolling(window=5, center=False).mean().shift(3).values
    
    # Alligator aligned: bullish when Lips > Teeth > Jaw
    bullish = (lips > teeth) & (teeth > jaw)
    bearish = (lips < teeth) & (teeth < jaw)
    
    bullish_aligned = align_htf_to_ltf(prices, df_12h, bullish)
    bearish_aligned = align_htf_to_ltf(prices, df_12h, bearish)
    
    # Volume confirmation: 20-period volume average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(bullish_aligned[i]) or 
            np.isnan(bearish_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Alligator bullish + volume above average
            if bullish_aligned[i] and volume[i] > 1.2 * vol_ma20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + volume above average
            elif bearish_aligned[i] and volume[i] > 1.2 * vol_ma20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish or volume drops significantly
            if bearish_aligned[i] or volume[i] < 0.7 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish or volume drops significantly
            if bullish_aligned[i] or volume[i] < 0.7 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals