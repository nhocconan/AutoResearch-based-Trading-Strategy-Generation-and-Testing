#!/usr/bin/env python3
"""
4h_1d_Donchian_Breakout_Volume_Trend_Filter
Hypothesis: Breakouts above/below 4h Donchian channels with volume confirmation and 1d trend filter.
Works in both bull and bear markets by only taking long when 1d close > 1d EMA50 and short when 1d close < 1d EMA50.
Volume confirmation ensures breakouts have conviction. Target: 20-40 trades/year on 4h (80-160 total over 4 years).
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
    
    # 4h Donchian channel (20-period high/low)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(lookback, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above Donchian high with volume expansion and 1d bullish trend
        long_condition = (high[i] > highest_high[i]) and volume_expansion[i] and (close[i] > ema50_1d_aligned[i])
        
        # Short: breakdown below Donchian low with volume expansion and 1d bearish trend
        short_condition = (low[i] < lowest_low[i]) and volume_expansion[i] and (close[i] < ema50_1d_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif long_condition and position == 1:
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_condition and position == -1:
            signals[i] = -position_size
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0