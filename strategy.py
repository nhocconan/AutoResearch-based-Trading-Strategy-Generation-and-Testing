#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data once (HTF)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Supertrend (ATR=10, mult=3)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    hl2 = (high_12h + low_12h) / 2
    upper = hl2 + 3 * atr
    lower = hl2 - 3 * atr
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_12h)
    direction = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > upper[i-1]:
            direction[i] = 1
        elif close_12h[i] < lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower[i] < lower[i-1]:
                lower[i] = lower[i-1]
            if direction[i] == -1 and upper[i] > upper[i-1]:
                upper[i] = upper[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower[i]
        else:
            supertrend[i] = upper[i]
    
    # Align Supertrend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Volume spike filter (20-period average on 4h data)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        st = supertrend_aligned[i]
        direction = direction_aligned[i]
        
        if position == 0:
            # Long: price above Supertrend + uptrend + volume spike
            if price > st and direction == 1 and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price below Supertrend + downtrend + volume spike
            elif price < st and direction == -1 and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses Supertrend in opposite direction
            if position == 1 and price < st:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Supertrend_10_3_12h_Volume_Spike"
timeframe = "4h"
leverage = 1.0