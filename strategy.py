#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Choppiness Index regime filter and Donchian(20) breakout.
# Long: Price breaks above Donchian high(20) + Choppiness Index < 38.2 (trending regime).
# Short: Price breaks below Donchian low(20) + Choppiness Index < 38.2 (trending regime).
# Uses 1d Choppiness Index to filter for trending markets, avoiding whipsaws in ranging conditions.
# Position size: 0.25 (25%) to balance risk and return.
# Target: 20-50 trades/year (80-200 over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ADX-like components for Choppiness Index
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr_period = 14
    atr = np.full(len(close_1d), np.nan)
    for i in range(atr_period, len(close_1d)):
        atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
    
    # Sum of True Range over period
    sum_tr = np.full(len(close_1d), np.nan)
    for i in range(atr_period, len(close_1d)):
        sum_tr[i] = np.nansum(tr[i-atr_period+1:i+1])
    
    # Max and min close over period
    max_hh = np.full(len(close_1d), np.nan)
    min_ll = np.full(len(close_1d), np.nan)
    for i in range(atr_period, len(close_1d)):
        max_hh[i] = np.nanmax(high_1d[i-atr_period+1:i+1])
        min_ll[i] = np.nanmin(low_1d[i-atr_period+1:i+1])
    
    # Choppiness Index: 100 * log10(sum_tr / (max_hh - min_ll)) / log10(atr_period)
    chop = np.full(len(close_1d), np.nan)
    for i in range(atr_period, len(close_1d)):
        if max_hh[i] > min_ll[i]:  # Avoid division by zero
            chop[i] = 100 * np.log10(sum_tr[i] / (max_hh[i] - min_ll[i])) / np.log10(atr_period)
    
    # Donchian channels (20-period) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Align 1d Choppiness Index to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        chop_val = chop_aligned[i]
        
        # Trending regime filter: Choppiness Index < 38.2
        trending_regime = chop_val < 38.2
        
        if position == 0:
            # Long: price breaks above Donchian high + trending regime
            if price > upper and trending_regime:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low + trending regime
            elif price < lower and trending_regime:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below Donchian low
            if price < lower:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above Donchian high
            if price > upper:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Chop_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0