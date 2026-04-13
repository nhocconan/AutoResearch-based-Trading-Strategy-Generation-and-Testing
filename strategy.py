#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w Donchian breakout + volume confirmation + 1w EMA trend filter
# Long when price breaks above 1w Donchian upper channel with volume > 1.5x average and price > 1w EMA50
# Short when price breaks below 1w Donchian lower channel with volume > 1.5x average and price < 1w EMA50
# Exit on opposite Donchian breakout or when price crosses 1w EMA50
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag and maximize edge
# Uses 1w timeframe for structure to reduce noise and false breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 150:  # Need sufficient data for 1w calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channels and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period)
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1w indicators to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(150, n):
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

name = "1d_1w_donchian_ema50_volume_filter_v1"
timeframe = "1d"
leverage = 1.0