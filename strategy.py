#!/usr/bin/env python3
name = "12h_WeeklyDonchianBreakout_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend: Donchian breakout on weekly high/low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period Donchian channels on weekly data
    high_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    donchian_low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Volume spike: volume > 2.0 * 20-period SMA of volume (on 12h)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_sma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # For Donchian and volume SMA
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_20w_aligned[i]) or np.isnan(donchian_low_20w_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + volume spike
            if close[i] > donchian_high_20w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low + volume spike
            elif close[i] < donchian_low_20w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly Donchian low
            if close[i] < donchian_low_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly Donchian high
            if close[i] > donchian_high_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals