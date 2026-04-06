#!/usr/bin/env python3
"""
6h Donchian breakout with 1d volume confirmation and ATR filter.
Hypothesis: Breakouts above/below Donchian(20) on 6h with 1d volume spike (volume > 2x 50-day average) and ATR volatility filter capture strong moves in both bull and bear markets, while avoiding choppy periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14291_6h_donchian20_1d_vol_atr_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = 0
    low_close[0] = 0
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return pd.Series(tr).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for volume and ATR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume 50-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d ATR(14)
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align to 6h timeframe (shifted by 1 day for completed bars only)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 6h volume > 2x 1d volume average
    vol_confirm = volume > (2 * vol_ma_1d_aligned)
    
    # ATR filter: avoid extremely low volatility (chop)
    # Calculate 6h ATR(14)
    atr_6h = calculate_atr(high, low, close, 14)
    atr_ma_6h = pd.Series(atr_6h).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_6h > (0.5 * atr_ma_6h)  # Only trade when volatility is above half the average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20, 50)
    start = max(20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or \
           np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or \
           np.isnan(atr_ma_6h[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to opposite Donchian band or ATR-based stop
        if position == 1:  # long position
            # Exit if price touches lower Donchian band or 2*ATR stop
            if close[i] <= low_min[i] or close[i] < entry_price - 2 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit if price touches upper Donchian band or 2*ATR stop
            if close[i] >= high_max[i] or close[i] > entry_price + 2 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with volume and volatility filter
            long_breakout = (close[i] > high_max[i]) and vol_confirm[i] and vol_filter[i]
            short_breakout = (close[i] < low_min[i]) and vol_confirm[i] and vol_filter[i]
            
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