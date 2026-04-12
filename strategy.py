#!/usr/bin/env python3
"""
6h_12h_Turtle_Squeeze_Breakout
Hypothesis: Trade Bollinger Band squeeze breakouts on 6h with 12h Donchian(20) as trend filter.
In Bollinger squeeze (low volatility), price often breaks out strongly. Use 12h Donchian to filter direction:
- Long when 6h breaks above Bollinger Upper Band AND 12h price > Donchian High(20)
- Short when 6h breaks below Bollinger Lower Band AND 12h price < Donchian Low(20)
Requires volume > 1.5x 20-period average to confirm breakout.
Works in both bull and bear markets: squeeze breakouts capture volatility expansion after consolidation.
Designed for low trade frequency (~15-30/year) with high conviction exits on opposite Donchian touch.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_Turtle_Squeeze_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H DATA FOR DONCHIAN TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # === 6H INDICATORS ===
    # Bollinger Bands (20, 2.0)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + (2.0 * std20)
    lower_bb = sma20 - (2.0 * std20)
    
    # Bollinger Band width for squeeze detection (optional filter)
    bb_width = (upper_bb - lower_bb) / sma20
    # Squeeze condition: BB width below 20-period average (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if not ready
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Long: 6h breaks above upper BB AND 12h price above Donchian High (uptrend)
        long_signal = (close[i] > upper_bb[i] and 
                      close[i] > donchian_high_aligned[i] and 
                      strong_volume)
        
        # Short: 6h breaks below lower BB AND 12h price below Donchian Low (downtrend)
        short_signal = (close[i] < lower_bb[i] and 
                       close[i] < donchian_low_aligned[i] and 
                       strong_volume)
        
        # Exit: opposite Donchian touch (turtle-style exit)
        exit_long = (position == 1 and close[i] < donchian_low_aligned[i])
        exit_short = (position == -1 and close[i] > donchian_high_aligned[i])
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals