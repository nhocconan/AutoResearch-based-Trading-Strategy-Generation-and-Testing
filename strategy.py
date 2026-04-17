#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Supertrend trend filter + 1w Donchian channel breakout + volume confirmation.
Long when price breaks above weekly Donchian(20) upper band with 1d Supertrend uptrend and volume > 1.3x 20-period 1d volume average.
Short when price breaks below weekly Donchian(20) lower band with 1d Supertrend downtrend and volume > 1.3x 20-period 1d volume average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
Weekly Donchian channels provide structural breakout levels; Supertrend filters for trending markets only; volume confirms participation.
Designed to work in bull markets (breakout continuation) and bear markets (strong trend continuation).
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
    
    # Get 1d data for Supertrend trend and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1d ATR (10-period) for Supertrend
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1d Supertrend (10, 3.0)
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if np.isnan(supertrend[i-1]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            supertrend[i] = supertrend[i-1] if not np.isnan(supertrend[i-1]) else upper_band[i]
            direction[i] = direction[i-1] if not np.isnan(direction[i-1]) else 1
            continue
            
        if close_1d[i] <= supertrend[i-1]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            supertrend[i] = lower_band[i]
            direction[i] = 1
            
        # Adjust bands
        if direction[i] == direction[i-1]:
            if direction[i] == 1:  # uptrend
                supertrend[i] = max(supertrend[i], lower_band[i])
            else:  # downtrend
                supertrend[i] = min(supertrend[i], upper_band[i])
        else:
            # Trend change
            if direction[i] == 1:  # changed to uptrend
                supertrend[i] = upper_band[i]
            else:  # changed to downtrend
                supertrend[i] = lower_band[i]
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly Donchian channels (20-period)
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align all to 12h
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Supertrend and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(direction_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_1d_aligned[i]
        # Trend filter: direction from Supertrend (1 for uptrend, -1 for downtrend)
        uptrend = direction_aligned[i] == 1
        downtrend = direction_aligned[i] == -1
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper band with uptrend and volume
            if (close[i] > donchian_upper_aligned[i] and 
                uptrend and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower band with downtrend and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  downtrend and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Donchian lower band
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Donchian upper band
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dSupertrend_1wDonchian20_Volume_Confirm"
timeframe = "12h"
leverage = 1.0