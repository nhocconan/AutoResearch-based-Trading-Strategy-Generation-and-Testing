#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Trend
Strategy: 4-hour breakout of daily Camarilla R1/S1 with volume confirmation and 12h trend filter.
Long: Price breaks above daily R1 + volume > 1.8x 20-period avg + price above 12h EMA34
Short: Price breaks below daily S1 + volume > 1.8x 20-period avg + price below 12h EMA34
Exit: Price returns to opposite Camarilla level (S1 for long, R1 for short)
Position size: 0.25
Designed to capture intraday breakouts with institutional levels in both bull and bear markets.
Timeframe: 4h
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
    
    # Calculate daily Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1
    r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_series_12h = pd.Series(close_12h)
    ema34_12h = close_series_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Trend filter: price above/below 12h EMA34
        price_above_ema = close[i] > ema34_12h_aligned[i]
        price_below_ema = close[i] < ema34_12h_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i-1]  # break above previous day R1
        breakout_down = close[i] < s1_aligned[i-1]  # break below previous day S1
        
        # Exit conditions: return to opposite Camarilla level
        return_to_s1 = close[i] < s1_aligned[i]  # for long positions
        return_to_r1 = close[i] > r1_aligned[i]  # for short positions
        
        if position == 0:
            # Long: breakout above R1 + volume filter + price above 12h EMA
            if breakout_up and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + price below 12h EMA
            elif breakout_down and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to S1 or break below S1
            if return_to_s1 or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to R1 or break above R1
            if return_to_r1 or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0