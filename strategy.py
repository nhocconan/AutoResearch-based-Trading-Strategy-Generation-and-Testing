#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Choppiness Index regime filter and Donchian channel breakout.
# Long: Price breaks above Donchian(20) high + Choppiness Index < 38.2 (trending regime).
# Short: Price breaks below Donchian(20) low + Choppiness Index < 38.2 (trending regime).
# Exit: Price crosses back through Donchian(20) midpoint.
# Uses 1d Choppiness Index to filter for trending markets only, reducing whipsaws in ranging periods.
# Position size: 0.25 to balance risk and return.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

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
    
    # Calculate True Range and ATR(14) for Choppiness Index
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    atr = np.full(len(close_1d), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr[i] = np.nanmean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate Choppiness Index: 100 * log10(sum(ATR)/ (max(high)-min(low))) / log10(period)
    chop = np.full(len(close_1d), np.nan)
    lookback = 14
    for i in range(lookback, len(close_1d)):
        sum_atr = np.nansum(atr[i-lookback+1:i+1])
        max_high = np.nanmax(high_1d[i-lookback+1:i+1])
        min_low = np.nanmin(low_1d[i-lookback+1:i+1])
        if max_high > min_low and sum_atr > 0:
            chop[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(lookback)
    
    # Align 1d Choppiness Index to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channel (20-period) on 12h
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Donchian midpoint for exit
    donch_mid = (donch_high + donch_low) / 2
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(donch_mid[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        chop_val = chop_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        mid = donch_mid[i]
        
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
            # Exit long: price crosses below Donchian midpoint
            if price < mid:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint
            if price > mid:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Chop_Trend"
timeframe = "12h"
leverage = 1.0