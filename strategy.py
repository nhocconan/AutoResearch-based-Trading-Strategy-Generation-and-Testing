#!/usr/bin/env python3
"""
1d_Weekly_Channel_Breakout_Volume
Hypothesis: Price breaking out of weekly Donchian channels with volume confirmation and weekly trend alignment captures strong moves in both bull and bear markets while limiting trades to avoid fee drag.
Long when price > weekly high (lookback 2) + volume > 1.5x 20-period average + weekly close > weekly EMA34.
Short when price < weekly low (lookback 2) + volume > 1.5x 20-period average + weekly close < weekly EMA34.
Exit on opposite signal or trend reversal. Position size: ±0.25.
Uses 1d for entry/exit and 1w for trend filter and breakout levels.
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
    
    # Volume confirmation (20-period MA on 1d)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter and breakout levels
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA34 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1w rolling high (2 periods) and low (2 periods)
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    high_2_1w = high_1w_series.rolling(window=2, min_periods=2).max().values
    low_2_1w = low_1w_series.rolling(window=2, min_periods=2).min().values
    
    # Align 1w high/low to 1d timeframe
    high_2_1w_aligned = align_htf_to_ltf(prices, df_1w, high_2_1w)
    low_2_1w_aligned = align_htf_to_ltf(prices, df_1w, low_2_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 2, 34)  # volume MA20, 1w high/low lookback, EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(high_2_1w_aligned[i]) or 
            np.isnan(low_2_1w_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: price > 1w high (2) + volume filter + 1w uptrend
            if close[i] > high_2_1w_aligned[i] and volume_filter and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < 1w low (2) + volume filter + 1w downtrend
            elif close[i] < low_2_1w_aligned[i] and volume_filter and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < 1w low (2) or 1w trend turns down
            if close[i] < low_2_1w_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > 1w high (2) or 1w trend turns up
            if close[i] > high_2_1w_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Channel_Breakout_Volume"
timeframe = "1d"
leverage = 1.0