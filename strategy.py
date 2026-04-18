#!/usr/bin/env python3
"""
12h_Donchian_Breakout_W1EMA34_Volume
Hypothesis: Price breaks above/below weekly Donchian(20) with volume spike and weekly EMA34 trend filter.
Captures major trend continuations with tight entry conditions to minimize fee drag.
Target: 15-30 trades/year to survive bear markets.
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
    
    # Weekly Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high/low (20-period)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h: weekly levels available after weekly bar closes
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Weekly EMA34 trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: >2.0x 50-period average (higher threshold for lower frequency)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 35)  # Warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        ema34 = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume spike and uptrend
            if price > donch_high and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume spike and downtrend
            elif price < donch_low and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below weekly EMA34 OR reverses below midpoint
            midpoint = (donch_high + donch_low) / 2.0
            if price < ema34:
                signals[i] = 0.0
                position = 0
            elif price < midpoint:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above weekly EMA34 OR reverses above midpoint
            midpoint = (donch_high + donch_low) / 2.0
            if price > ema34:
                signals[i] = 0.0
                position = 0
            elif price > midpoint:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_W1EMA34_Volume"
timeframe = "12h"
leverage = 1.0