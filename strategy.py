#!/usr/bin/env python3
"""
4h_Donchian_20_Breakout_Volume_Trend_1d
Hypothesis: Combines 4-hour Donchian(20) breakouts with daily trend filter (price vs daily EMA200) and volume confirmation (1.5x 48-bar average volume) to capture high-probability trend continuations. Designed for low trade frequency (15-30 per year) to minimize fee drag while capturing strong directional moves. Works in both bull and bear by following daily trend direction.
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
    
    # Get daily data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4-hour Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 48-period MA (8 days of 4h bars)
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 48  # Wait for volume MA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA200
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_48[i])
        
        # Breakout conditions at Donchian channels
        long_breakout = close[i] > high_20[i] and vol_confirm and uptrend
        short_breakout = close[i] < low_20[i] and vol_confirm and downtrend
        
        # Exit conditions: return to opposite Donchian level
        long_exit = close[i] < low_20[i]
        short_exit = close[i] > high_20[i]
        
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_20_Breakout_Volume_Trend_1d"
timeframe = "4h"
leverage = 1.0