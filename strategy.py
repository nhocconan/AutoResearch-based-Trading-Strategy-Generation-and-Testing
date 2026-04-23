#!/usr/bin/env python3
"""
Hypothesis: Daily close above/below weekly Donchian channels with volume confirmation.
Long when daily close breaks above weekly Donchian high (20-period) and volume > 1.5x average.
Short when daily close breaks below weekly Donchian low and volume > 1.5x average.
Exit when price returns to weekly Donchian midpoint or volume dries up.
Designed for low frequency (~10-25 trades/year) to capture major trends while avoiding whipsaws.
Works in both bull and breakout markets by trading breakouts with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channels - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high and low
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max()
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min()
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high.values)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low.values)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid.values)
    
    # Volume confirmation: 20-day average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close breaks above weekly Donchian high with volume confirmation
            if close[i] > donchian_high_aligned[i] and volume[i] > 1.5 * vol_avg[i]:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below weekly Donchian low with volume confirmation
            elif close[i] < donchian_low_aligned[i] and volume[i] > 1.5 * vol_avg[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to weekly Donchian midpoint or low volume
                if close[i] <= donchian_mid_aligned[i] or volume[i] < 0.5 * vol_avg[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to weekly Donchian midpoint or low volume
                if close[i] >= donchian_mid_aligned[i] or volume[i] < 0.5 * vol_avg[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_WeeklyDonchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0