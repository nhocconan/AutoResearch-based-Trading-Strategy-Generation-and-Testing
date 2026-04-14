#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour strategy using 1-day Supertrend (ATR=10, mult=3) for trend filter and 12-hour Donchian channel breakout for entries.
# In trending markets (Supertrend bullish), buy breakouts above upper Donchian(20).
# In bearish trends (Supertrend bearish), sell breakdowns below lower Donchian(20).
# Volume > 1.5x 20-period average confirms momentum.
# ATR-based stop loss: exit when price moves against position by 2.5x ATR.
# Target: 12-37 trades/year per symbol (48-148 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the trend via Supertrend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Supertrend (ATR=10, mult=3) for trend direction
    atr_len = 10
    mult = 3
    if len(df_1d) < atr_len:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR
    atr_1d = pd.Series(tr).ewm(span=atr_len, adjust=False, min_periods=atr_len).mean().values
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upperband = hl2 + (mult * atr_1d)
    lowerband = hl2 - (mult * atr_1d)
    
    # Initialize Supertrend arrays
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if np.isnan(atr_1d[i]) or np.isnan(upperband[i]) or np.isnan(lowerband[i]):
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
        else:
            if close_1d[i] > supertrend[i-1]:
                direction[i] = 1
            elif close_1d[i] < supertrend[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
            
            if direction[i] == 1:
                supertrend[i] = max(lowerband[i], supertrend[i-1])
            else:
                supertrend[i] = min(upperband[i], supertrend[i-1])
    
    # Supertrend direction: 1 = uptrend, -1 = downtrend
    supertrend_direction = direction
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1d, supertrend_direction)
    
    # 12-hour Donchian channel (20 periods)
    dc_len = 20
    if n < dc_len:
        return np.zeros(n)
    
    # Calculate Donchian channels
    highest_high = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().values
    lowest_low = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss (12-hour ATR)
    atr_len_stop = 10
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_12h = pd.Series(tr).ewm(span=atr_len_stop, adjust=False, min_periods=atr_len_stop).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(dc_len, atr_len_stop*2, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakout/breakdown in direction of 1d Supertrend
            if supertrend_direction_aligned[i] == 1:  # Uptrend
                # Buy breakout above upper Donchian
                if (close[i] > highest_high[i] and 
                    volume_confirmed):
                    position = 1
                    signals[i] = position_size
            elif supertrend_direction_aligned[i] == -1:  # Downtrend
                # Sell breakdown below lower Donchian
                if (close[i] < lowest_low[i] and 
                    volume_confirmed):
                    position = -1
                    signals[i] = -position_size
        elif position == 1:
            # Exit long: price closes below lower Donchian or hits ATR-based stop
            # Stop loss: 2.5 * ATR below highest high since entry (simplified: use current ATR)
            if (close[i] < lowest_low[i] or 
                close[i] < (high[i] - 2.5 * atr_12h[i])):  # Simplified stop
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper Donchian or hits ATR-based stop
            if (close[i] > highest_high[i] or 
                close[i] > (low[i] + 2.5 * atr_12h[i])):  # Simplified stop
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Supertrend_Donchian_Breakout_v1"
timeframe = "12h"
leverage = 1.0