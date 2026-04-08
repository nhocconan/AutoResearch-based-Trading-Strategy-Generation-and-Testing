#!/usr/bin/env python3
# 4h_12h_Donchian_Breakout_Volume_Confirmation_v1
# Hypothesis: Breakout of 12-hour Donchian channels with volume confirmation on 4h timeframe.
# Uses 12h Donchian breakout as primary signal with volume confirmation to filter false breakouts.
# Works in both bull and bear markets by following the higher timeframe trend structure.
# Target: 15-40 trades/year (60-160 total over 4 years) to avoid fee drag.

name = "4h_12h_Donchian_Breakout_Volume_Confirmation_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 12-hour data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: 20-period high
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (wait for 12h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation on 4h: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_confirm = vol_ratio > 1.8
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if Donchian levels are not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]):
            if position != 0:
                # Hold position until exit
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Only consider new signals during session with volume confirmation
        if not (in_session[i] and vol_confirm[i]):
            if position != 0:
                # Hold existing position
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 12h Donchian low (breakdown)
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h Donchian high (breakout)
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 12h Donchian high with volume
            if close[i] > donchian_high_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 12h Donchian low with volume
            elif close[i] < donchian_low_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals