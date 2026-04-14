#!/usr/bin/env python3
"""
4h Donchian Breakout + 1W Trend + Volume Spike
Long when price breaks above 20-period Donchian high, weekly close > weekly open, and volume > 1.5x average.
Short when price breaks below 20-period Donchian low, weekly close < weekly open, and volume > 1.5x average.
Exit when price reverses back through the Donchian median.
Designed for low turnover: ~15-30 trades/year per symbol.
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
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Calculate 20-period Donchian channels
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume filter: 20-period average
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count >= 20:
            vol_ma[i] = vol_sum / vol_count
            vol_sum -= volume[i - 19]
            vol_count -= 1
    
    # Weekly trend: 1 if bullish (close > open), -1 if bearish (close < open)
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Create arrays for alignment
    weekly_bullish_arr = weekly_bullish.astype(float)
    weekly_bearish_arr = weekly_bearish.astype(float)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any indicator not ready
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get aligned weekly trend values
        weekly_bull = align_htf_to_ltf(prices, df_1w, weekly_bullish_arr)[i]
        weekly_bear = align_htf_to_ltf(prices, df_1w, weekly_bearish_arr)[i]
        
        if np.isnan(weekly_bull) or np.isnan(weekly_bear):
            continue
        
        if position == 0:
            # Long: Break above Donchian high, volume spike, weekly bullish
            if close[i] > donch_high[i] and volume[i] > vol_ma[i] * 1.5 and weekly_bull > 0.5:
                position = 1
                signals[i] = position_size
            # Short: Break below Donchian low, volume spike, weekly bearish
            elif close[i] < donch_low[i] and volume[i] > vol_ma[i] * 1.5 and weekly_bear > 0.5:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price crosses below Donchian midpoint
            if close[i] < donch_mid[i] and close[i-1] >= donch_mid[i-1]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price crosses above Donchian midpoint
            if close[i] > donch_mid[i] and close[i-1] <= donch_mid[i-1]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0