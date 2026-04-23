#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h Supertrend trend filter and volume confirmation.
- Long: Close > DonchianH20 AND Supertrend(12h) = uptrend AND volume > 1.5x 20-period avg
- Short: Close < DonchianL20 AND Supertrend(12h) = downtrend AND volume > 1.5x 20-period avg
- Exit: Opposite Donchian breakout OR Supertrend flip
- Uses 12h HTF for Supertrend to avoid whipsaw in ranging markets
- Designed for low trade frequency (12-37/year) to minimize fee drag on 6h timeframe
- Works in bull (buy breakouts above H20 in uptrend) and bear (sell breakdowns below L20 in downtrend)
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_h = high_roll
    donchian_l = low_roll
    
    # Calculate 12h Supertrend for trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR for 12h
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h)
    tr2 = pd.Series(high_12h) - pd.Series(close_12h).shift(1)
    tr3 = pd.Series(low_12h) - pd.Series(close_12h).shift(1)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate Supertrend
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + multiplier * atr_12h
    lower_band = hl2 - multiplier * atr_12h
    
    # Initialize Supertrend arrays
    supertrend = np.full_like(close_12h, np.nan)
    direction = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Start calculation after warmup period
    start_idx = atr_period
    for i in range(start_idx, len(close_12h)):
        if np.isnan(atr_12h[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            continue
            
        if i == start_idx:
            # Initialize
            supertrend[i] = upper_band[i]
            direction[i] = 1  # start with uptrend assumption
        else:
            prev_close = close_12h[i-1]
            prev_supertrend = supertrend[i-1]
            prev_direction = direction[i-1]
            
            if prev_direction == 1:
                # Was in uptrend
                supertrend[i] = max(lower_band[i], prev_supertrend)
                if close_12h[i] < supertrend[i]:
                    # Flip to downtrend
                    direction[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    direction[i] = 1
            else:
                # Was in downtrend
                supertrend[i] = min(upper_band[i], prev_supertrend)
                if close_12h[i] > supertrend[i]:
                    # Flip to uptrend
                    direction[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    direction[i] = -1
    
    # Align Supertrend direction to 6h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, atr_period)  # Need 20 for Donchian, 10 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_h[i]) or
            np.isnan(donchian_l[i]) or
            np.isnan(supertrend_dir_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout signals (using current close vs current bands)
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        if position == 0:
            # Long: Donchian H20 breakout up AND Supertrend uptrend AND volume confirmation
            if breakout_up and supertrend_dir_aligned[i] == 1 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian L20 breakout down AND Supertrend downtrend AND volume confirmation
            elif breakout_down and supertrend_dir_aligned[i] == -1 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian L20 breakout down OR Supertrend flip to downtrend
            if breakout_down or supertrend_dir_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian H20 breakout up OR Supertrend flip to uptrend
            if breakout_up or supertrend_dir_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hSupertrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0