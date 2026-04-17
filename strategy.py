#!/usr/bin/env python3
"""
4h_TripleConfirmation_Breakout
Strategy: 4-hour breakout with triple confirmation - price channel breakout, volume surge, and 1-day trend alignment.
Long: Price breaks above Donchian high(20) + volume > 2x 20-period average + price above 1-day EMA50
Short: Price breaks below Donchian low(20) + volume > 2x 20-period average + price below 1-day EMA50
Exit: Price returns to Donchian midpoint OR opposite breakout
Position size: 0.25
Designed for high-conviction trades with low frequency to minimize fee drag in both bull and bear markets.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_rolling + low_rolling) / 2
    
    # Calculate 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1-day EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_rolling[i]) or 
            np.isnan(low_rolling[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period average (strict filter for fewer trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below 1-day EMA50
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        # Breakout conditions (using previous period's channel to avoid look-ahead)
        breakout_up = close[i] > high_rolling[i-1]   # break above previous period high
        breakout_down = close[i] < low_rolling[i-1]  # break below previous period low
        
        # Return to midpoint (exit condition)
        return_to_mid = abs(close[i] - donchian_mid[i]) < 0.15 * (high_rolling[i] - low_rolling[i])
        
        if position == 0:
            # Long: breakout up + volume filter + price above EMA50
            if breakout_up and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: breakout down + volume filter + price below EMA50
            elif breakout_down and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to midpoint OR break down (contrarian signal)
            if return_to_mid or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to midpoint OR break up (contrarian signal)
            if return_to_mid or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TripleConfirmation_Breakout"
timeframe = "4h"
leverage = 1.0