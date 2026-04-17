#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Williams %R filter + 4h Donchian(20) breakout + volume confirmation.
Long when price breaks above 4h Donchian(20) high with 12h Williams %R < -80 (oversold) and volume > 1.5x 20-period volume average.
Short when price breaks below 4h Donchian(20) low with 12h Williams %R > -20 (overbought) and volume > 1.5x 20-period volume average.
Williams %R on 12h timeframe helps identify overextended moves in the primary trend, increasing probability of continuation breakouts.
Designed to work in bull markets (breakout from oversold) and bear markets (breakdown from overbought).
Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing (0.25) to minimize fee drag.
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
    
    # Get 12h data for Williams %R
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Williams %R (14-period)
    def williams_r(high_vals, low_vals, close_vals, window):
        highest_high = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close_vals) / (highest_high - lowest_low)
        # Handle division by zero when highest_high == lowest_low
        wr = np.where((highest_high - lowest_low) == 0, -50, wr)
        return wr
    
    wr_14_12h = williams_r(high_12h, low_12h, close_12h, 14)
    
    # Calculate 4h Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # Calculate 4h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h Williams %R to 4h timeframe
    wr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, wr_14_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_14_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian(20) high with 12h oversold and volume
            if (close[i] > donchian_upper[i] and 
                wr_14_12h_aligned[i] < -80 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian(20) low with 12h overbought and volume
            elif (close[i] < donchian_lower[i] and 
                  wr_14_12h_aligned[i] > -20 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 4h Donchian(20) low (opposite side of channel)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 4h Donchian(20) high (opposite side of channel)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hWilliamsR14_Donchian20_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0