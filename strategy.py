#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day Donchian breakout + volume confirmation + 1-day EMA trend filter
# Long when price breaks above 1-day Donchian upper channel with volume > 1.5x average and price > 1-day EMA50
# Short when price breaks below 1-day Donchian lower channel with volume > 1.5x average and price < 1-day EMA50
# Exit on opposite Donchian breakout or when price crosses 1-day EMA50
# Uses daily timeframe for structure to reduce noise and false breakouts
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1d indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume surge condition
        volume_surge = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # EMA trend filter
        above_ema = close[i] > ema50_aligned[i]
        below_ema = close[i] < ema50_aligned[i]
        
        # Entry logic: breakout + volume + trend filter
        long_entry = long_breakout and volume_surge and above_ema
        short_entry = short_breakout and volume_surge and below_ema
        
        # Exit conditions: opposite breakout or EMA cross
        exit_long = position == 1 and (short_breakout or close[i] < ema50_aligned[i])
        exit_short = position == -1 and (long_breakout or close[i] > ema50_aligned[i])
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_ema50_volume_filter_v2"
timeframe = "4h"
leverage = 1.0