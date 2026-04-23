#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h Supertrend(ATR=10,mult=3) trend filter and volume confirmation.
- Long: Close > Donchian Upper(20) AND Supertrend=uptrend AND volume > 1.5x 20-period avg
- Short: Close < Donchian Lower(20) AND Supertrend=downtrend AND volume > 1.5x 20-period avg
- Exit: Opposite Donchian breakout OR Supertrend flip
- Uses 12h HTF for Supertrend to avoid whipsaw in ranging markets
- Designed for low trade frequency (20-50/year) to minimize fee drag
- Works in bull (buy breakouts above upper band) and bear (sell breakdowns below lower band)
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
    
    # Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h Supertrend for trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range and ATR(10) for Supertrend
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First period has no previous close
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    # Initialize Supertrend arrays
    supertrend = np.zeros_like(close_12h)
    direction = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    # Set first value
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    # Calculate Supertrend
    for i in range(1, len(close_12h)):
        if close_12h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_12h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = upper_band[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = lower_band[i]
        elif direction[i] == 1:
            supertrend[i] = max(upper_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(lower_band[i], supertrend[i-1])
    
    # Align Supertrend direction to 4h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(supertrend_direction_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # Close above prior upper band
        breakout_down = close[i] < donchian_lower[i-1]  # Close below prior lower band
        
        if position == 0:
            # Long: Donchian upper breakout AND Supertrend uptrend AND volume confirmation
            if breakout_up and supertrend_direction_aligned[i] == 1 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian lower breakout AND Supertrend downtrend AND volume confirmation
            elif breakout_down and supertrend_direction_aligned[i] == -1 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian lower breakout OR Supertrend downtrend
            if breakout_down or supertrend_direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian upper breakout OR Supertrend uptrend
            if breakout_up or supertrend_direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hSupertrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0