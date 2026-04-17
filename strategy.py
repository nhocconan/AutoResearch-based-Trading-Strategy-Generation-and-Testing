#!/usr/bin/env python3
"""
12h_Donchian20_VolumeSpike_TrendFilter_v1
Long: Price breaks above Donchian(20) high + volume spike + trend filter (price > 1d EMA50)
Short: Price breaks below Donchian(20) low + volume spike + trend filter (price < 1d EMA50)
Exit: Price crosses back below Donchian(20) mid (for long) or above mid (for short)
Position size: 0.25
Designed to capture breakouts with volume confirmation in trending markets.
Target: 20-50 total trades over 4 years (5-12/year) to avoid fee drag.
Works in both bull (breakouts continue) and bear (failed reversals) regimes.
"""

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
    
    # === Donchian Channel (20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # === Volume Spike (2x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(vol_spike[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: breakout above Donchian high + volume spike + above 1d EMA50
            if (close[i] > donch_high[i] and 
                vol_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: breakout below Donchian low + volume spike + below 1d EMA50
            elif (close[i] < donch_low[i] and 
                  vol_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses back below Donchian mid
            if close[i] < donch_mid[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above Donchian mid
            if close[i] > donch_mid[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0