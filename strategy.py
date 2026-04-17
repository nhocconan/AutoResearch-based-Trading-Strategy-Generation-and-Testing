#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w EMA200 trend filter + 1d Donchian(20) breakout + volume confirmation.
Long when price breaks above 1d Donchian upper channel with 1w EMA200 rising and volume > 1.5x 20-period 1d volume average.
Short when price breaks below 1d Donchian lower channel with 1w EMA200 falling and volume > 1.5x 20-period 1d volume average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 30-100 total trades over 4 years.
Weekly EMA200 filters major trend alignment; Donchian breakout captures momentum; volume confirms participation.
Designed to work in bull markets (trend-following breakouts) and bear markets (mean reversion after volatility expansion via short signals).
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
    
    # Get 1d data for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for EMA200 trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA200
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_prev = np.roll(ema_200_1w, 1)
    ema_200_1w_prev[0] = np.nan
    ema_200_1w_rising = ema_200_1w > ema_200_1w_prev
    ema_200_1w_falling = ema_200_1w < ema_200_1w_prev
    
    # Align all to 1d (primary timeframe)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    ema_200_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w_rising)
    ema_200_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w_falling)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for weekly EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(ema_200_1w_rising_aligned[i]) or np.isnan(ema_200_1w_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with rising 1w EMA200 and volume
            if (close[i] > donchian_upper_aligned[i] and 
                ema_200_1w_rising_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with falling 1w EMA200 and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema_200_1w_falling_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Donchian middle (or use lower for tighter stop)
            donchian_middle = (donchian_upper + donchian_lower) / 2
            donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Donchian middle
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wEMA200_Donchian20_Breakout_Volume_Confirm"
timeframe = "1d"
leverage = 1.0