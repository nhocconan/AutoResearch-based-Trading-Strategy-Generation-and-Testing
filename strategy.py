#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian(20) breakout with 1-week EMA20 trend filter and volume confirmation.
Long when price breaks above 1-day upper Donchian channel with 1-week EMA20 rising and volume spike.
Short when price breaks below 1-day lower Donchian channel with 1-week EMA20 falling and volume spike.
Exit when price crosses the 1-day 20-period EMA.
Designed for low trade frequency by requiring multiple confirmations and using daily trend filter.
Works in both bull and bear markets by following the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Donchian channels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper and lower bands
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (each day's levels apply to the entire day)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Load 1-week data for EMA20 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1-week EMA20
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to 1d timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Exit condition: 1-day 20-period EMA
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for EMA20
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(ema20_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian with 1-week EMA20 rising and volume spike
            if (close[i] > upper_20_aligned[i] and 
                ema20_1w_aligned[i] > ema20_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with 1-week EMA20 falling and volume spike
            elif (close[i] < lower_20_aligned[i] and 
                  ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses the 1-day 20-period EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below EMA20
                if close[i] < ema20_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above EMA20
                if close[i] > ema20_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_20_1wEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0