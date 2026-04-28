#!/usr/bin/env python3
# Hypothesis: 12h Donchian channel breakout with 1-day ATR filter and volume confirmation.
# Uses Donchian(20) breakouts as primary signal, filtered by 1-day ATR expansion (volatility regime)
# and volume > 1.5x 20-period average. Designed for 12h timeframe to target 50-150 total trades
# over 4 years (12-37/year). Works in both bull and bear markets by capturing breakouts
# during volatility expansions while avoiding false signals in low-volatility periods.

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
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR calculation using Wilder's smoothing
    atr = np.full_like(tr, np.nan, dtype=float)
    atr[13] = np.mean(tr[:14])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate ATR ratio: current ATR / 20-period average ATR
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / atr_ma
    
    # Align ATR ratio to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian channel (20-period) on 12h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR ratio > 1.2 indicates volatility expansion
        volatility_expansion = atr_ratio_aligned[i] > 1.2
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above upper band
        breakout_down = close[i] < donchian_low[i-1]  # Break below lower band
        
        # Entry conditions with volume confirmation
        long_entry = volatility_expansion and breakout_up and volume_filter[i]
        short_entry = volatility_expansion and breakout_down and volume_filter[i]
        
        # Exit conditions: when volatility contracts or opposite breakout
        volatility_contracts = atr_ratio_aligned[i] < 0.8
        long_exit = volatility_contracts or breakout_down or (position == 1 and close[i] < donchian_low[i-1])
        short_exit = volatility_contracts or breakout_up or (position == -1 and close[i] > donchian_high[i-1])
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_1dATR_VolumeFilter"
timeframe = "12h"
leverage = 1.0