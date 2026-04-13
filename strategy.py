#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h Donchian channel breakout + volume confirmation + session filter
# Long when price breaks above 4h Donchian high (20-period) and volume > 1.5x average
# Short when price breaks below 4h Donchian low (20-period) and volume > 1.5x average
# Trade only during 08:00-20:00 UTC to avoid low-liquidity hours
# Use 4h for trend direction and structure, 1h only for entry timing precision
# Target: 60-150 total trades over 4 years = 15-37/year to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channel (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # Pre-compute session hours
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% of capital
    
    for i in range(20, n):
        # Skip if data not ready or outside session
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            not (8 <= hours[i] <= 20)):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume confirmation
        breakout_up = close[i] > donchian_high_aligned[i] and volume[i] > 1.5 * vol_ma[i]
        breakout_down = close[i] < donchian_low_aligned[i] and volume[i] > 1.5 * vol_ma[i]
        
        # Exit on opposite breakout
        exit_long = position == 1 and breakout_down
        exit_short = position == -1 and breakout_up
        
        # Execute signals
        if breakout_up and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_down and position != -1:
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

name = "1h_4h_donchian_breakout_volume_session"
timeframe = "1h"
leverage = 1.0