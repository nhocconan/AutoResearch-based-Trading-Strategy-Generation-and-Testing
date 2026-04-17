#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h Donchian channel breakout + volume confirmation + session filter.
Long when price breaks above 4h Donchian upper (20) with volume > 1.5x 20-period average and hour in 08-20 UTC.
Short when price breaks below 4h Donchian lower (20) with volume confirmation and hour in 08-20 UTC.
Uses 4h structure for direction, 1h for precise entry timing, volume to avoid false breakouts,
and session filter to reduce noise during low-liquidity hours. Designed for range-bound 2025 market.
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
    
    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channel (20-period)
    # Upper = highest high over 20 periods
    # Lower = lowest low over 20 periods
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper with volume confirmation
            if (close[i] > donchian_upper_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower with volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 4h Donchian lower (opposite side)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above 4h Donchian upper (opposite side)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian20_Breakout_Volume_Confirmation_Session"
timeframe = "1h"
leverage = 1.0