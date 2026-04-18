#!/usr/bin/env python3
"""
1d_Donchian_Breakout_Volume_Trend_v1
Strategy: Daily Donchian channel breakout with volume confirmation and weekly trend filter.
Long: Close breaks above 20-day Donchian upper band + volume > 1.5x average + weekly close > weekly EMA34.
Short: Close breaks below 20-day Donchian lower band + volume > 1.5x average + weekly close < weekly EMA34.
Exit: Opposite Donchian band touch.
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
Uses volume to filter false breakouts and weekly trend to align with higher timeframe momentum.
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
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-day Donchian channels
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all data to daily timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        if position == 0:
            # Long: close above upper Donchian + volume + weekly uptrend
            if close[i] > upper_20_aligned[i] and vol_confirm and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below lower Donchian + volume + weekly downtrend
            elif close[i] < lower_20_aligned[i] and vol_confirm and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close touches lower Donchian band
            if close[i] < lower_20_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close touches upper Donchian band
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_Volume_Trend_v1"
timeframe = "1d"
leverage = 1.0