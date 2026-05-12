#!/usr/bin/env python3
"""
4h Donchian Breakout + 12h EMA Trend + Volume Confirmation
Hypothesis: Donchian breakouts capture breakout momentum, while 12h EMA filters direction.
Volume confirmation ensures institutional participation. Works in bull/bear via trend filter.
Target: 20-40 trades/year to minimize fee drag.
"""
name = "4h_Donchian_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h EMA Trend ===
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === Volume Spike (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close above Donchian High + price above 12h EMA + volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_12h_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below Donchian Low + price below 12h EMA + volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_12h_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Close below Donchian Low
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above Donchian High
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals