#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for 52-week high/low
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 52-week high/low
    highest_52w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    lowest_52w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    
    # Align 52-week levels to 6h timeframe
    highest_52w_6h = align_htf_to_ltf(prices, df_1w, highest_52w)
    lowest_52w_6h = align_htf_to_ltf(prices, df_1w, lowest_52w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume for daily confirmation
    avg_vol_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily volume average to 6h timeframe
    avg_vol_20d_6h = align_htf_to_ltf(prices, df_1d, avg_vol_20d)
    
    # Calculate 6h ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 52  # Need 52-week high/low and 20-day volume average
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_52w_6h[i]) or 
            np.isnan(lowest_52w_6h[i]) or 
            np.isnan(avg_vol_20d_6h[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current daily volume > 1.5 * 20-day average
        daily_volume_index = i // 4  # Approximate 6h to daily index (4x 6h = 1 day)
        if daily_volume_index >= len(volume_1d):
            signals[i] = 0.0
            continue
            
        volume_filter = volume_1d[daily_volume_index] > (1.5 * avg_vol_20d[daily_volume_index])
        
        # Volatility filter: ATR > 0 (always true but keeps structure)
        volatility_filter = atr[i] > 0
        
        if position == 0:
            # Long: price breaks above 52-week high with volume confirmation
            if close[i] > highest_52w_6h[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 52-week low with volume confirmation
            elif close[i] < lowest_52w_6h[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below 52-week high or volume drops
            if close[i] < highest_52w_6h[i] or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above 52-week low or volume drops
            if close[i] > lowest_52w_6h[i] or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_52Week_High_Low_Breakout_Volume"
timeframe = "6h"
leverage = 1.0