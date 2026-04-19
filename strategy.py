#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h Supertrend (ATR=10, mult=3) for trend direction,
# 4h Donchian(20) breakout for entry timing, and volume confirmation.
# Only take long in uptrend (price > Supertrend) when breaking above Donchian high,
# only short in downtrend (price < Supertrend) when breaking below Donchian low.
# Volume filter ensures breakout strength. Designed to work in both bull and bear
# markets by following the trend and catching breakouts in trend direction.
# Target: 60-120 total trades over 4 years (15-30/year).
name = "4h_12h_Supertrend_Donchian20_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend calculation (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR for Supertrend (12h timeframe)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_period = 10
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < atr_period:
            atr[i] = np.nan
        elif i == atr_period:
            atr[i] = np.nanmean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend calculation
    upper_band = (high_12h + low_12h) / 2 + 3 * atr
    lower_band = (high_12h + low_12h) / 2 - 3 * atr
    supertrend = np.full_like(close_12h, np.nan)
    direction = np.full_like(close_12h, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(atr[i-1]):
            supertrend[i] = np.nan
            continue
        if close_12h[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_12h[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend and direction to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # 4h Donchian channels (20-period high/low)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(volume_ma[i]) or np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long when: uptrend (direction=1) AND price breaks above Donchian high WITH volume
            if direction_aligned[i] == 1 and close[i] > donch_high[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when: downtrend (direction=-1) AND price breaks below Donchian low WITH volume
            elif direction_aligned[i] == -1 and close[i] < donch_low[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when trend changes to downtrend OR price breaks below Donchian low
            if direction_aligned[i] == -1 or close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when trend changes to uptrend OR price breaks above Donchian high
            if direction_aligned[i] == 1 or close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals