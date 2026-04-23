#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Supertrend for trend direction and 1h Donchian breakout for entry timing.
- Long when: 4h Supertrend is bullish AND price breaks above 1h Donchian upper channel (20) AND volume > 1.5x 20-period average
- Short when: 4h Supertrend is bearish AND price breaks below 1h Donchian lower channel (20) AND volume > 1.5x 20-period average
- Exit when: price crosses the Donchian middle (mean of upper/lower) OR 4h Supertrend flips
- Uses 4h Supertrend as trend filter to avoid counter-trend trades
- Volume confirmation reduces false breakouts
- Designed for both bull and bear markets: Supertrend adapts to regime
- Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe to minimize fee drag
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
    
    # Calculate ATR(10) for Supertrend
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation (4h)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR for Supertrend (4h)
    tr1_4h = np.abs(high_4h[1:] - low_4h[1:])
    tr2_4h = np.abs(high_4h[1:] - close_4h[:-1])
    tr3_4h = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend upper/lower bands
    hl2_4h = (high_4h + low_4h) / 2
    upper_4h = hl2_4h + 3.0 * atr_4h
    lower_4h = hl2_4h - 3.0 * atr_4h
    
    # Supertrend trend calculation
    supertrend_4h = np.full_like(close_4h, np.nan)
    direction_4h = np.ones_like(close_4h)  # 1 for up, -1 for down
    
    for i in range(10, len(close_4h)):
        if np.isnan(upper_4h[i-1]) or np.isnan(lower_4h[i-1]):
            continue
            
        # Upper band
        if close_4h[i-1] <= upper_4h[i-1]:
            upper_4h[i] = upper_4h[i]
        else:
            upper_4h[i] = hl2_4h[i] + 3.0 * atr_4h[i]
            
        # Lower band
        if close_4h[i-1] >= lower_4h[i-1]:
            lower_4h[i] = lower_4h[i]
        else:
            lower_4h[i] = hl2_4h[i] - 3.0 * atr_4h[i]
            
        # Trend
        if close_4h[i] <= supertrend_4h[i-1]:
            direction_4h[i] = -1
        else:
            direction_4h[i] = 1
            
        if direction_4h[i] == 1 and direction_4h[i-1] == -1:
            supertrend_4h[i] = lower_4h[i]
        elif direction_4h[i] == -1 and direction_4h[i-1] == 1:
            supertrend_4h[i] = upper_4h[i]
        else:
            supertrend_4h[i] = supertrend_4h[i-1] if direction_4h[i] == direction_4h[i-1] else (lower_4h[i] if direction_4h[i] == 1 else upper_4h[i])
    
    # Align Supertrend direction to 1h
    supertrend_dir_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # Donchian channels (1h)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_middle = (highest_high + lowest_low) / 2
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 10)  # Need 20 for Donchian, 20 for volume MA, 10 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(supertrend_dir_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above previous period's upper channel
        breakout_down = close[i] < lowest_low[i-1]  # Break below previous period's lower channel
        
        # Volume confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: 4h Supertrend bullish + Donchian breakout up + volume spike
            if supertrend_dir_4h_aligned[i] == 1 and breakout_up and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: 4h Supertrend bearish + Donchian breakout down + volume spike
            elif supertrend_dir_4h_aligned[i] == -1 and breakout_down and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit conditions:
            # 1. Price crosses Donchian middle (mean reversion)
            # 2. 4h Supertrend flips bearish
            exit_middle = close[i] < donchian_middle[i]
            exit_trend = supertrend_dir_4h_aligned[i] == -1
            
            if exit_middle or exit_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit conditions:
            # 1. Price crosses Donchian middle (mean reversion)
            # 2. 4h Supertrend flips bullish
            exit_middle = close[i] > donchian_middle[i]
            exit_trend = supertrend_dir_4h_aligned[i] == 1
            
            if exit_middle or exit_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Supertrend4h_Donchian20_VolumeSpike"
timeframe = "1h"
leverage = 1.0