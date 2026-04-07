#!/usr/bin/env python3
"""
4h_ma_crossover_1d_trend_volume_v1
Hypothesis: On 4h timeframe, use EMA crossover (8/21) for entry signals, filtered by 1d EMA50 trend and volume confirmation.
- In uptrend (price > 1d EMA50): long when EMA8 crosses above EMA21, exit when EMA8 crosses below EMA21 or trend reverses
- In downtrend (price < 1d EMA50): short when EMA8 crosses below EMA21, exit when EMA8 crosses above EMA21 or trend reverses
Volume confirms genuine momentum. This strategy captures trends in both bull and bear markets while avoiding whipsaws in ranging conditions.
Target: 20-50 trades/year (~80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ma_crossover_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50 = df_1d['close'].ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # EMA crossover on 4h
    ema_fast = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate crossover signals
    ema_crossover = np.zeros(n)
    ema_crossover[ema_fast > ema_slow] = 1   # bullish
    ema_crossover[ema_fast < ema_slow] = -1  # bearish
    
    # Volume confirmation (20-period average on 4h = ~3.3 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_crossover[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2x average volume
        vol_confirm = volume[i] > 1.2 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: EMA8 crosses below EMA21 or trend turns bearish
            if ema_crossover[i] == -1 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: EMA8 crosses above EMA21 or trend turns bullish
            if ema_crossover[i] == 1 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: EMA8 crosses above EMA21 with volume in uptrend
            if (ema_crossover[i] == 1 and 
                vol_confirm and 
                close[i] > ema_50_aligned[i]):  # uptrend filter
                position = 1
                signals[i] = 0.25
            # Short entry: EMA8 crosses below EMA21 with volume in downtrend
            elif (ema_crossover[i] == -1 and 
                  vol_confirm and 
                  close[i] < ema_50_aligned[i]):  # downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals