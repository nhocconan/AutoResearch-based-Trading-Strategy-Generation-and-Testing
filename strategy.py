#!/usr/bin/env python3
"""
6h Donchian Breakout + Weekly Trend Filter + Volume Confirmation
Long: Price breaks above Donchian(20) high AND weekly close > weekly open (bullish week) AND volume > 1.5x avg volume
Short: Price breaks below Donchian(20) low AND weekly close < weekly open (bearish week) AND volume > 1.5x avg volume
Exit: Opposite Donchian break (long exits on lower band break, short exits on upper band break)
Designed to capture strong momentum moves with weekly trend alignment and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Weekly Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # Bullish week = close > open
    weekly_bearish = weekly_close < weekly_open  # Bearish week = close < open
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # === Volume Confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume[i]) or np.isnan(volume_threshold[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian lower band
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper band
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Break above upper band + bullish week + volume surge
            if (close[i] > highest_high[i] and 
                weekly_bullish_aligned[i] and 
                volume[i] > volume_threshold[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Break below lower band + bearish week + volume surge
            elif (close[i] < lowest_low[i] and 
                  weekly_bearish_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                position = -1
                signals[i] = -0.25
    
    return signals