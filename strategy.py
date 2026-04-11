#!/usr/bin/env python3
# 6h_12h_donchian_volume_breakout_v1
# Strategy: 6h Donchian breakout with 12h volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Donchian(20) breakouts on 6h are filtered by 12h volume surge (>1.5x average)
# to avoid false breakouts. Works in both bull (breakouts continuation) and bear (breakdowns).
# Target: 20-50 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_donchian_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 60-period volume average on 12h (equivalent to ~30 periods on 6h)
    vol_avg_60 = pd.Series(df_12h['volume']).rolling(window=60, min_periods=60).mean().values
    vol_avg_60_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_60)
    
    # 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_avg_60_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 60-period average
        # Need to get current 12h volume - use the aligned 12h volume data
        vol_12h_current = align_htf_to_ltf(prices, df_12h, df_12h['volume'].values)[i]
        vol_confirm = vol_12h_current > 1.5 * vol_avg_60_aligned[i]
        
        # Breakout signals
        breakout_up = high[i] > high_20[i-1]
        breakdown_down = low[i] < low_20[i-1]
        
        # Entry conditions
        if breakout_up and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakdown_down and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Donchian break
        elif position == 1 and low[i] < low_20[i-1]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > high_20[i-1]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals