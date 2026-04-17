#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h Donchian breakout (20-period) + 1d EMA50 trend filter + volume confirmation.
Long when price breaks above 4h Donchian upper channel with 1d EMA50 rising and volume > 1.5x 20-period 1h volume average.
Short when price breaks below 4h Donchian lower channel with 1d EMA50 falling and volume > 1.5x 20-period 1h volume average.
Uses discrete position sizing 0.20 to limit fee drag. Target: 60-150 total trades over 4 years (15-37/year).
Session filter (08-20 UTC) to reduce noise trades. Uses 4h for signal direction, 1h only for entry timing.
Designed to capture medium-term trends while avoiding whipsaws in ranging markets.
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
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = np.nan
    ema_50_1d_rising = ema_50_1d > ema_50_1d_prev
    ema_50_1d_falling = ema_50_1d < ema_50_1d_prev
    
    # Get 1h data for volume confirmation
    vol_ma_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 1h
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_1d_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_rising)
    ema_50_1d_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_falling)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, 20)  # need enough for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_1d_rising_aligned[i]) or 
            np.isnan(ema_50_1d_falling_aligned[i]) or np.isnan(vol_ma_20_1h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1h[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper with rising 1d EMA50 and volume
            if (close[i] > donchian_upper_aligned[i] and 
                ema_50_1d_rising_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower with falling 1d EMA50 and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema_50_1d_falling_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 4h Donchian middle (or lower for tighter stop)
            donchian_middle = (donchian_upper + donchian_lower) / 2
            donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above 4h Donchian middle
            donchian_middle = (donchian_upper + donchian_lower) / 2
            donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian20_1dEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0