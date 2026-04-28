#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Trend_1d
Hypothesis: Combines Donchian channel breakout with 1d EMA trend filter and volume confirmation.
Targets 20-40 trades/year by requiring breakout + trend alignment + volume surge to reduce false signals.
Designed to work in both bull and bear markets via long/short symmetry and volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > high_20[i] and uptrend and volume_surge[i]
        bearish_breakout = close[i] < low_20[i] and downtrend and volume_surge[i]
        
        # Exit conditions: opposite breakout with volume
        bullish_exit = close[i] < low_20[i] and volume_surge[i]
        bearish_exit = close[i] > high_20[i] and volume_surge[i]
        
        if bullish_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif bearish_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif bullish_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif bearish_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Trend_1d"
timeframe = "4h"
leverage = 1.0