#!/usr/bin/env python3
# 6h_12h_ema_crossover_volume_filter_v1
# Strategy: 6h EMA crossover (9/21) filtered by 12h volume surge (>2x average)
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: EMA crossovers on 6h capture medium-term trends, but are prone to whipsaws.
# Filtering by 12h volume spikes ensures trades occur only during high conviction moves,
# reducing false signals in both bull and bear markets. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_ema_crossover_volume_filter_v1"
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
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period volume average on 12h (equivalent to ~25 periods on 6h)
    vol_avg_50 = pd.Series(df_12h['volume']).rolling(window=50, min_periods=50).mean().values
    vol_avg_50_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_50)
    
    # 6h EMA crossover (9/21)
    close_series = pd.Series(close)
    ema9 = close_series.ewm(span=9, min_periods=9, adjust=False).mean().values
    ema21 = close_series.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or 
            np.isnan(vol_avg_50_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get current 12h volume (aligned)
        vol_12h_current = align_htf_to_ltf(prices, df_12h, df_12h['volume'].values)[i]
        vol_confirm = vol_12h_current > 2.0 * vol_avg_50_aligned[i]
        
        # EMA crossover signals
        ema_cross_up = ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]
        ema_cross_down = ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1]
        
        # Entry conditions: EMA crossover + volume confirmation
        if ema_cross_up and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif ema_cross_down and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite EMA crossover
        elif position == 1 and ema_cross_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and ema_cross_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals