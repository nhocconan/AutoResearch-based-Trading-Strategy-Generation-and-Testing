#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Donchian(20) breakout on 4h with 1-week EMA50 trend filter and volume confirmation.
Works in bull/bear markets: In strong weekly trends (price > weekly EMA50 for longs, < weekly EMA50 for shorts),
Donchian(20) breakouts with volume capture momentum while avoiding counter-trend whipsaws.
Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
Targets 20-50 trades over 4 years on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels from 4h data (using completed bars only)
    # We calculate on 4h data but align the signals properly
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position
    
    # Warmup: need 20-period Donchian, 1w EMA50, vol avg
    start_idx = max(21, 50, 20)  # +1 for shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        highest_val = highest_20[i]
        lowest_val = lowest_20[i]
        ema_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with weekly trend alignment and volume spike
            long_condition = (close_val > highest_val and 
                            close_val > ema_val and 
                            vol_spike)
            short_condition = (close_val < lowest_val and 
                             close_val < ema_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price re-enters Donchian channel (below midpoint) OR loses weekly trend alignment
            midpoint = (highest_val + lowest_val) * 0.5
            if close_val < midpoint or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters Donchian channel (above midpoint) OR loses weekly trend alignment
            midpoint = (highest_val + lowest_val) * 0.5
            if close_val > midpoint or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0