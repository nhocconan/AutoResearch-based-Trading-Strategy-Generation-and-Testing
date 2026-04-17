#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above 20-day high AND 1w EMA34 is rising AND 1d volume > 1.5x 20-day average volume.
Short when price breaks below 20-day low AND 1w EMA34 is falling AND 1d volume > 1.5x 20-day average volume.
Exit when price returns to the 10-day midpoint (mean reversion) or opposite breakout occurs.
Uses 1w for trend direction, 1d for breakout and volume filters. Designed to capture strong trends
with volume confirmation while avoiding choppy markets. Target: 15-25 trades/year per symbol.
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
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-day Donchian channels on 1d
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-day midpoint for exit
    midpoint_10 = (pd.Series(high_1d).rolling(window=10, min_periods=10).max().values + 
                   pd.Series(low_1d).rolling(window=10, min_periods=10).min().values) / 2
    
    # Calculate 1w EMA34 and its slope (rising/falling)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_prev = np.roll(ema_34_1w, 1)
    ema_34_1w_prev[0] = np.nan
    ema_rising = ema_34_1w > ema_34_1w_prev
    ema_falling = ema_34_1w < ema_34_1w_prev
    
    # Calculate 20-day average volume on 1d
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 1h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    midpoint_10_aligned = align_htf_to_ltf(prices, df_1d, midpoint_10)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Align 1w EMA34 and its slope to 1h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or
            np.isnan(midpoint_10_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        # Get the aligned 1d volume for this timestamp
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmed = not np.isnan(volume_1d_aligned[i]) and \
                          not np.isnan(vol_ma_20_1d_aligned[i]) and \
                          volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > high_20_aligned[i]
        breakout_down = close[i] < low_20_aligned[i]
        
        # Mean reversion exit condition
        revert_to_midpoint = abs(close[i] - midpoint_10_aligned[i]) < 0.001 * close[i]  # within 0.1%
        
        # Opposite breakout exit condition
        opposite_breakout = (position == 1 and breakout_down) or (position == -1 and breakout_up)
        
        if position == 0:
            # Long: breakout above 20-day high with rising 1w EMA34 and volume confirmation
            if (breakout_up and ema_rising_aligned[i] == 1.0 and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: breakout below 20-day low with falling 1w EMA34 and volume confirmation
            elif (breakout_down and ema_falling_aligned[i] == 1.0 and volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 10-day midpoint OR opposite breakout
            if (revert_to_midpoint or opposite_breakout):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 10-day midpoint OR opposite breakout
            if (revert_to_midpoint or opposite_breakout):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume_Regime"
timeframe = "1d"
leverage = 1.0