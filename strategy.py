#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
Long when price breaks above 20-day high and weekly EMA10 is rising.
Short when price breaks below 20-day low and weekly EMA10 is falling.
Exit when price crosses opposite 20-day boundary or weekly EMA trend reverses.
Daily timeframe ensures low trade frequency; weekly EMA filters trend to avoid counter-trend trades.
Works in both bull and bear markets by following weekly trend while using daily Donchian for entries.
Volume confirmation reduces false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian channels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly data for EMA10 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Daily Donchian channels (20-period)
    high_20d = pd.Series(close_1d).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(close_1d).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA10 for trend filter
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Daily volume average (20-period) for confirmation
    vol_avg_20d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20d[i]) or np.isnan(low_20d[i]) or np.isnan(ema10_1w_aligned[i]) or
            np.isnan(vol_avg_20d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-day high, weekly EMA10 rising, volume above average
            if (close[i] > high_20d[i] and 
                ema10_1w_aligned[i] > ema10_1w_aligned[i-1] and
                volume[i] > vol_avg_20d[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low, weekly EMA10 falling, volume above average
            elif (close[i] < low_20d[i] and 
                  ema10_1w_aligned[i] < ema10_1w_aligned[i-1] and
                  volume[i] > vol_avg_20d[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below 20-day low OR weekly EMA10 turns down
                if (close[i] < low_20d[i] or 
                    ema10_1w_aligned[i] < ema10_1w_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above 20-day high OR weekly EMA10 turns up
                if (close[i] > high_20d[i] or 
                    ema10_1w_aligned[i] > ema10_1w_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_Donchian20_WeeklyEMA10_Volume"
timeframe = "1d"
leverage = 1.0