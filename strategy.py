#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper band (20) AND 4h EMA20 rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band (20) AND 4h EMA20 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Donchian channel (between upper and lower bands).
# Uses 4h for direction and 1h for entry timing to reduce false breakouts.
# Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_Donchian_20_4hEMA20_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low) from 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper band (20-period high)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band (20-period low)
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 4h EMA20 for trend filter
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 4h EMA20 direction
    ema20_rising = np.zeros_like(ema20_4h_aligned, dtype=bool)
    ema20_falling = np.zeros_like(ema20_4h_aligned, dtype=bool)
    ema20_rising[1:] = ema20_4h_aligned[1:] > ema20_4h_aligned[:-1]
    ema20_falling[1:] = ema20_4h_aligned[1:] < ema20_4h_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Sufficient warmup for EMA20 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or np.isnan(ema20_rising[i]) or 
            np.isnan(ema20_falling[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, 4h EMA20 rising, volume filter
            long_cond = (close[i] > donchian_high_aligned[i]) and ema20_rising[i] and volume_filter[i]
            # Short conditions: price breaks below Donchian lower band, 4h EMA20 falling, volume filter
            short_cond = (close[i] < donchian_low_aligned[i]) and ema20_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses back above Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals