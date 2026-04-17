#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_R1S1_Breakout_Volume_Filter
Strategy: Hourly breakout of daily Camarilla R1/S1 levels with volume confirmation and 4h trend filter.
Long: Price breaks above daily R1 + volume > 1.5x 20-period avg + price above 4h EMA20
Short: Price breaks below daily S1 + volume > 1.5x 20-period avg + price below 4h EMA20
Exit: Price returns to opposite Camarilla level (S1 for long, R1 for short) or 4h EMA20
Position size: 0.20
Designed to capture intraday breakouts aligned with 4h trend in both bull and bear markets.
Timeframe: 1h
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
    
    # Camarilla R1 and S1: close + (high-low)*1.1/12 and close - (high-low)*1.1/12
    camarilla_width = (high_1d - low_1d) * 1.1 / 12.0
    r1_1d = close_1d + camarilla_width
    s1_1d = close_1d - camarilla_width
    
    # Calculate 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    close_series_4h = pd.Series(close_4h)
    ema20_4h = close_series_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily levels and 4h EMA to 1h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume confirmation (20-period MA on 1h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Ensure warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 4h EMA20
        price_above_ema = close[i] > ema20_4h_aligned[i]
        price_below_ema = close[i] < ema20_4h_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_1d_aligned[i-1]  # break above previous day R1
        breakout_down = close[i] < s1_1d_aligned[i-1]  # break below previous day S1
        
        # Exit conditions: return to opposite level or cross 4h EMA20
        return_to_s1 = close[i] < s1_1d_aligned[i]  # for long exit
        return_to_r1 = close[i] > r1_1d_aligned[i]  # for short exit
        cross_ema_down = close[i] < ema20_4h_aligned[i]  # for long exit
        cross_ema_up = close[i] > ema20_4h_aligned[i]  # for short exit
        
        if position == 0:
            # Long: breakout up + volume filter + price above 4h EMA20
            if breakout_up and volume_filter and price_above_ema:
                signals[i] = 0.20
                position = 1
            # Short: breakout down + volume filter + price below 4h EMA20
            elif breakout_down and volume_filter and price_below_ema:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: return to S1 or cross below 4h EMA20
            if return_to_s1 or cross_ema_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: return to R1 or cross above 4h EMA20
            if return_to_r1 or cross_ema_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_Camarilla_R1S1_Breakout_Volume_Filter"
timeframe = "1h"
leverage = 1.0