#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day trend and volume confirmation.
Long when price breaks above Donchian(20), 1-day EMA34 rising, and volume > 1.5x avg.
Short when price breaks below Donchian(20), 1-day EMA34 falling, and volume > 1.5x avg.
Exit when price crosses opposite Donchian boundary.
Donchian provides clear breakout levels; 1-day EMA34 filters trend; volume confirms strength.
Designed for low trade frequency by requiring all three conditions simultaneously.
Works in both bull and bear markets by following daily trend while using 4h breakout for entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_avg[i]) or
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian, EMA34 rising, volume > 1.5x avg
            if (close[i] > high_20[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian, EMA34 falling, volume > 1.5x avg
            elif (close[i] < low_20[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and
                  volume[i] > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below lower Donchian
                if close[i] < low_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above upper Donchian
                if close[i] > high_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0