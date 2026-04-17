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
    
    # Get weekly data for ATR-based volatility filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR(14) for volatility regime filter
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.maximum(np.abs(low_1w[1:] - close_1w[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get daily data for price channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-day) on daily
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align weekly ATR and daily channels to daily timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Volatility filter: only trade when weekly ATR is above its 50-period average
    atr_ma50 = pd.Series(atr_1w_aligned).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_1w_aligned > atr_ma50
    
    # Volume filter: current volume > 1.5 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * volume_ma20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for ATR MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(atr_ma50[i]) or
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 20-day high with volume and volatility filter
            if (close[i] > highest_20_aligned[i] and volume_filter[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below 20-day low with volume and volatility filter
            elif (close[i] < lowest_20_aligned[i] and volume_filter[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below 20-day low
            if close[i] < lowest_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above 20-day high
            if close[i] > highest_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Volume_VolatilityFilter"
timeframe = "1d"
leverage = 1.0