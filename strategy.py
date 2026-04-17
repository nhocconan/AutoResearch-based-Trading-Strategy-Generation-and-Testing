#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Confirmation
Strategy: 12h Camarilla R1/S1 breakout with volume confirmation and 1d EMA34 trend filter.
Long: Close breaks above R1 + volume > 1.5x average + close above 1d EMA34
Short: Close breaks below S1 + volume > 1.5x average + close below 1d EMA34
Exit: Close crosses back below R1 (long) or above S1 (short)
Position size: 0.25
Designed to capture intraday breakouts aligned with daily trend.
Timeframe: 12h
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
    
    # Calculate daily pivot points (using previous day's HLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Calculate EMA34 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h[i]) or 
            np.isnan(s1_12h[i]) or 
            np.isnan(ema34_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema34_12h[i]
        price_below_ema = close[i] < ema34_12h[i]
        
        # Breakout conditions
        breakout_above_r1 = close[i] > r1_12h[i]
        breakdown_below_s1 = close[i] < s1_12h[i]
        
        if position == 0:
            # Long: Close breaks above R1 + volume filter + price above EMA
            if breakout_above_r1 and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 + volume filter + price below EMA
            elif breakdown_below_s1 and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close crosses back below R1
            if close[i] < r1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close crosses back above S1
            if close[i] > s1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0