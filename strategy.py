#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h Supertrend(ATR=10,mult=3) + volume confirmation
# Donchian breakout captures trend momentum; Supertrend filters counter-trend moves
# Volume confirmation ensures breakouts have institutional participation
# Works in both bull/bear: breakouts work in trends, Supertrend avoids false signals in ranges
# Target: 75-200 total trades over 4 years (19-50/year)
name = "4h_Donchian20_12hSupertrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Supertrend for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR for Supertrend
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_12h
    
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    upperband = (high_12h + low_12h) / 2 + 3 * atr
    lowerband = (high_12h + low_12h) / 2 - 3 * atr
    
    supertrend = np.full_like(close_12h, np.nan, dtype=float)
    direction = np.full_like(close_12h, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(atr[i-1]) or np.isnan(close_12h[i-1]):
            continue
            
        if close_12h[i] > upperband[i-1]:
            direction[i] = 1
        elif close_12h[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if direction[i] == -1 and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    
    # Donchian(20) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + above Supertrend + volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > supertrend_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below Supertrend + volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < supertrend_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low or below Supertrend
            if (close[i] < lowest_low[i]) or (close[i] < supertrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high or above Supertrend
            if (close[i] > highest_high[i]) or (close[i] > supertrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals