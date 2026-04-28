#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_Volume
Hypothesis: 4-hour Donchian channel breakouts with daily trend filter and volume confirmation. 
Targets 15-40 trades/year by requiring price to break above/below the 20-period high/low, 
aligned with the daily trend, and confirmed by volume surge. Works in both bull and bear 
markets by trading with the daily trend direction while using Donchian channels for 
objective breakout levels. Volume filter reduces false breakouts.
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
    
    # Get daily data for trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian channel (20-period high/low)
    # Using previous day's close to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate 20-period rolling high/low on daily data
    high_20 = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all higher timeframe data to 4h
    donchian_high = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_20)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for sufficient warmup (20 for Donchian + buffer)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price > EMA34 = bullish, < EMA34 = bearish
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions with trend alignment and volume surge
        # Long: price breaks above Donchian high + daily uptrend + volume surge
        long_entry = (close[i] > donchian_high[i] and 
                     trend_up and 
                     volume_surge[i])
        
        # Short: price breaks below Donchian low + daily downtrend + volume surge
        short_entry = (close[i] < donchian_low[i] and 
                      trend_down and 
                      volume_surge[i])
        
        # Exit on opposite break with volume surge
        long_exit = close[i] < donchian_low[i] and volume_surge[i]
        short_exit = close[i] > donchian_high[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
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

name = "4h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0