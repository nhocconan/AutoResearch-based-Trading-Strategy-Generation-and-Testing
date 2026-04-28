#!/usr/bin/env python3
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
    
    # Get 1d data for ATR-based volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align length
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 6h
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 6h ATR(14) for dynamic position sizing
    tr6_1 = high[1:] - low[1:]
    tr6_2 = np.abs(high[1:] - close[:-1])
    tr6_3 = np.abs(low[1:] - close[:-1])
    tr6 = np.maximum(np.maximum(tr6_1, tr6_2), tr6_3)
    tr6 = np.concatenate([[np.nan], tr6])
    atr_14_6h = pd.Series(tr6).rolling(window=14, min_periods=14).mean().values
    
    # Get weekly data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian(20)
    highest_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to 6h
    highest_20_1w_aligned = align_htf_to_ltf(prices, df_1w, highest_20_1w)
    lowest_20_1w_aligned = align_htf_to_ltf(prices, df_1w, lowest_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(highest_20_1w_aligned[i]) or 
            np.isnan(lowest_20_1w_aligned[i]) or
            np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when 1d ATR is above its 50-period median
        # This avoids choppy markets and focuses on volatile periods
        if i >= 50:
            atr_slice = atr_14_1d_aligned[max(0, i-49):i+1]
            atr_median = np.nanmedian(atr_slice)
            volatile_regime = not np.isnan(atr_median) and atr_14_1d_aligned[i] > atr_median
        else:
            volatile_regime = False
        
        # Dynamic position size based on 6h ATR (inverse volatility)
        # Size = 0.30 * (median ATR / current ATR) capped at 0.30
        if i >= 50:
            atr6_slice = atr_14_6h[max(0, i-49):i+1]
            atr6_median = np.nanmedian(atr6_slice)
            if not np.isnan(atr6_median) and atr_14_6h[i] > 0:
                vol_scalar = min(atr6_median / atr_14_6h[i], 1.0)
                base_size = 0.30 * vol_scalar
            else:
                base_size = 0.15
        else:
            base_size = 0.15
        
        # Entry conditions: Weekly Donchian breakout in volatile regime
        long_breakout = close[i] > highest_20_1w_aligned[i]
        short_breakout = close[i] < lowest_20_1w_aligned[i]
        
        long_entry = long_breakout and volatile_regime
        short_entry = short_breakout and volatile_regime
        
        # Exit conditions: Close crosses back below/above the opposite Donchian level
        long_exit = close[i] < lowest_20_1w_aligned[i]
        short_exit = close[i] > highest_20_1w_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = base_size
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -base_size
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = base_size
            elif position == -1:
                signals[i] = -base_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyDonchian20_VolatileRegime_v1"
timeframe = "6h"
leverage = 1.0