#!/usr/bin/env python3
"""
6h Donchian Breakout + 1d SuperTrend + Volume
Hypothesis: Donchian(20) breakouts with 1d SuperTrend filter and volume confirmation.
Captures strong momentum moves in both bull and bear markets by only trading in the
direction of the higher timeframe trend. Volume ensures breakout conviction.
Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14455_6h_donchian20_1d_supertrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for SuperTrend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # SuperTrend parameters
    atr_period = 10
    atr_multiplier = 3.0
    
    # Calculate ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_1d = pd.Series(tr_1d).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate upper and lower bands
    hl_avg = (high_1d + low_1d) / 2
    upper_band = hl_avg + (atr_multiplier * atr_1d)
    lower_band = hl_avg - (atr_multiplier * atr_1d)
    
    # Initialize SuperTrend
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1d)):
        if np.isnan(atr_1d[i]):
            continue
            
        # Update bands
        if close_1d[i-1] > upper_band[i-1]:
            upper_band[i] = max(upper_band[i], upper_band[i-1])
        else:
            upper_band[i] = upper_band[i]
            
        if close_1d[i-1] < lower_band[i-1]:
            lower_band[i] = min(lower_band[i], lower_band[i-1])
        else:
            lower_band[i] = lower_band[i]
        
        # Determine trend
        if close_1d[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_1d[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        # Set SuperTrend value
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align SuperTrend to 6h
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) on 6h
    donchian_period = 20
    highest = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require 1.5x average volume for breakout
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(donchian_period, atr_period) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or
            np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: Donchian breakdown OR SuperTrend turns bearish OR stoploss
            if (close[i] < lowest[i] or direction_aligned[i] == -1 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Donchian breakout OR SuperTrend turns bullish OR stoploss
            if (close[i] > highest[i] or direction_aligned[i] == 1 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + SuperTrend alignment + volume
            long_breakout = (close[i] > highest[i] and direction_aligned[i] == 1 and vol_filter[i])
            short_breakout = (close[i] < lowest[i] and direction_aligned[i] == -1 and vol_filter[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals