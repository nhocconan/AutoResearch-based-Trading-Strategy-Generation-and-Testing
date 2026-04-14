#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Spike + Weekly Trend Filter
Long when price breaks above 12h Donchian upper (20) with volume > 1.5x average and weekly close > weekly open.
Short when price breaks below 12h Donchian lower (20) with volume > 1.5x average and weekly close < weekly open.
Exit when price crosses Donchian midline (10-period average of upper/lower).
Designed for low turnover: ~15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, window):
    """Calculate Donchian upper, lower, and midline"""
    upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
    midline = (upper + lower) / 2
    return upper, lower, midline

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Calculate Donchian channels (20-period)
    upper, lower, midline = donchian_channels(high, low, 20)
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend: 1 if bullish (close > open), -1 if bearish (close < open)
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Create arrays for alignment
    weekly_bullish_arr = weekly_bullish.astype(float)
    weekly_bearish_arr = weekly_bearish.astype(float)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # Get aligned weekly trend values
        weekly_bull = align_htf_to_ltf(prices, df_1w, weekly_bullish_arr)[i]
        weekly_bear = align_htf_to_ltf(prices, df_1w, weekly_bearish_arr)[i]
        
        if np.isnan(weekly_bull) or np.isnan(weekly_bear):
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian, volume spike, weekly bullish
            if close[i] > upper[i] and volume[i] > vol_ma[i] * 1.5 and weekly_bull > 0.5:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower Donchian, volume spike, weekly bearish
            elif close[i] < lower[i] and volume[i] > vol_ma[i] * 1.5 and weekly_bear > 0.5:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price crosses below midline
            if close[i] < midline[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price crosses above midline
            if close[i] > midline[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0