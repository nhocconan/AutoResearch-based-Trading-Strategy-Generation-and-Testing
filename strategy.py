#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyTrend_VolumeSpike
Hypothesis: Donchian(20) breakout on 6h with weekly EMA50 trend filter and volume confirmation.
Designed for 6h timeframe targeting 80-180 total trades over 4 years.
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear markets:
In trending regimes (price > weekly EMA50 for longs, < weekly EMA50 for shorts),
Donchian breakouts with volume spike capture strong momentum continuations.
Exit on opposite Donchian breakout or trend reversal.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Donchian channels (20-period) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Discrete size to reduce fee churn
    
    # Warmup: need weekly EMA50, Donchian(20), vol avg
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        highest_val = highest_high[i]
        lowest_val = lowest_low[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with weekly EMA alignment and volume spike
            long_condition = (close_val > highest_val and 
                            close_val > ema_val and 
                            vol_spike)
            short_condition = (close_val < lowest_val and 
                             close_val < ema_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR weekly trend reversal
            if close_val < lowest_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR weekly trend reversal
            if close_val > highest_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0