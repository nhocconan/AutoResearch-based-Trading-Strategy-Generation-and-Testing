#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for weekly trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate weekly EMA200 on 1d
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 6h
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Donchian(20) on 6h (highest high and lowest low of last 20 bars)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike (24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Ensure EMA200 is ready
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high, above 1d EMA200, volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema200_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low, below 1d EMA200, volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema200_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below Donchian low or below 1d EMA200
            if close[i] < lowest_low[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above Donchian high or above 1d EMA200
            if close[i] > highest_high[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals