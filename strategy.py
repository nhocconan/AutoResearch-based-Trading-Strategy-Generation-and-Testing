#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with volume confirmation and chop filter
    # Long: price breaks above H3 + volume > 1.5x average + chop < 61.8 (trending)
    # Short: price breaks below L3 + volume > 1.5x average + chop < 61.8 (trending)
    # Exit: price crosses Camarilla pivot level (mean reversion)
    # Uses 1d Camarilla levels for structure, 4h for execution timing
    # Volume confirmation reduces false breakouts
    # Chop filter ensures we trade in trending markets only
    # Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fees
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # H3 = pivot + (range * 1.1 / 2)
    # L3 = pivot - (range * 1.1 / 2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h3_1d = pivot_1d + (range_1d * 1.1 / 2.0)
    l3_1d = pivot_1d - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 4h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Calculate 4h volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Choppiness Index (14-period)
    # Chop = 100 * log10(sum(atr) / (max(high) - min(low))) / log10(period)
    # We'll use a simplified version: chop = 100 * log10(atr_sum / (hh - ll)) / log10(period)
    # But for efficiency, we'll use a proxy: chop < 61.8 = trending, chop > 61.8 = ranging
    # We'll calculate true range and use it in chop calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) == 0, 50, chop)  # neutral when no range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for indicators
        # Skip if data not ready
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop filter: chop < 61.8 = trending market (good for breakouts)
        trending_market = chop[i] < 61.8
        
        # Breakout conditions
        long_breakout = close[i] > h3_1d_aligned[i] and close[i-1] <= h3_1d_aligned[i-1]
        short_breakout = close[i] < l3_1d_aligned[i] and close[i-1] >= l3_1d_aligned[i-1]
        
        # Exit conditions: price crosses pivot level (mean reversion)
        exit_long = position == 1 and close[i] < pivot_1d_aligned[i]
        exit_short = position == -1 and close[i] > pivot_1d_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and volume_confirmed and trending_market and position != 1
        short_entry = short_breakout and volume_confirmed and trending_market and position != -1
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0