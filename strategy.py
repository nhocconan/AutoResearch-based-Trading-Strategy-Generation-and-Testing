#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_1dTrend_Volume_Strategy"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and Donchian calculation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for long-term trend filter
    close_d = df_d['close'].values
    ema200_d = pd.Series(close_d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_d_aligned = align_htf_to_ltf(prices, df_d, ema200_d)
    
    # Calculate daily Donchian channels (20-period)
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    donchian_high = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_d, donchian_low)
    
    # Volume filter: current volume > 1.5 * 20-period average (using 12h volume)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough data for EMA200
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema200_d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema200_val = ema200_d_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price above Donchian high + above EMA200 + volume filter
            if close[i] > donchian_high_val and close[i] > ema200_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below Donchian low + below EMA200 + volume filter
            elif close[i] < donchian_low_val and close[i] < ema200_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below Donchian low or below EMA200
            if close[i] < donchian_low_val or close[i] < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above Donchian high or above EMA200
            if close[i] > donchian_high_val or close[i] > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals