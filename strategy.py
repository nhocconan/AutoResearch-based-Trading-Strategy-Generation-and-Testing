#!/usr/bin/env python3
"""
Hypothesis: 1d trend strategy using weekly Donchian breakout with volume confirmation.
Uses weekly Donchian(20) breakout for trend direction, confirmed by daily volume > 1.5x 20-day average.
Trades only in breakout direction (long on weekly high break, short on weekly low break).
Designed to capture strong trending moves while avoiding choppy markets with volume filter.
Target: 10-20 trades/year by requiring weekly breakout + volume confirmation.
Works in both bull and bear markets via breakout direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Donchian(20) for trend direction ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channels (20-period high/low)
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, high_max_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, low_min_20)
    
    # === Daily volume confirmation ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for calculations
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: today's volume > 1.5x 20-day average
        vol_today_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirmed = vol_today_aligned[i] > vol_ma_20_1d_aligned[i] * 1.5
        
        # Skip if volume not confirmed
        if not vol_confirmed:
            signals[i] = 0.0
            position = 0
            continue
        
        # Breakout signals
        high_break = close[i] > donchian_high_20_aligned[i]
        low_break = close[i] < donchian_low_20_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above weekly Donchian high
            if high_break:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below weekly Donchian low
            elif low_break:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: exit on opposite breakout
        elif position == 1:
            # Exit long if price breaks below weekly Donchian low
            if low_break:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above weekly Donchian high
            if high_break:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0