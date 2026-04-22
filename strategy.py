#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian breakout with 1-week trend filter and volume confirmation.
Long when price breaks above Donchian(20) high, weekly EMA50 rising, and volume > 1.5x average.
Short when price breaks below Donchian(20) low, weekly EMA50 falling, and volume > 1.5x average.
Exit when price crosses opposite Donchian boundary.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following weekly trend while using daily breakouts for entries.
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
    
    # Load 1-week data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
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
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, weekly EMA50 rising, volume confirmation
            if (close[i] > high_20[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, weekly EMA50 falling, volume confirmation
            elif (close[i] < low_20[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and
                  volume[i] > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below Donchian low
                if close[i] < low_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above Donchian high
                if close[i] > high_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0