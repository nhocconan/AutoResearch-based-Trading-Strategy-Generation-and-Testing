#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout (20-period) + 1d EMA50 trend filter + volume > 1.5x 20-period average.
Enter long when price breaks above upper band in uptrend (price > 1d EMA50), short when breaks below lower band in downtrend.
Uses Donchian for breakout signals, 1d EMA for trend direction, volume for confirmation.
Targets 20-30 trades/year per symbol (80-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned 1d EMA
        ema = ema_50_aligned[i]
        
        # Check for NaN values
        if np.isnan(ema) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i]):
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Long: price breaks above upper Donchian in uptrend
                if close[i] > high_20[i] and close[i-1] <= high_20[i-1] and close[i] > ema:
                    position = 1
                    signals[i] = position_size
                # Short: price breaks below lower Donchian in downtrend
                elif close[i] < low_20[i] and close[i-1] >= low_20[i-1] and close[i] < ema:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when price breaks below lower band
            if close[i] < low_20[i] and close[i-1] >= low_20[i-1]:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price breaks above upper band
            if close[i] > high_20[i] and close[i-1] <= high_20[i-1]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dEMA50_Volume1.5x"
timeframe = "4h"
leverage = 1.0