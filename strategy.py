#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_Trend
Hypothesis: Trades Donchian(20) breakouts in the direction of 1d trend (above/below 1d EMA50)
with volume confirmation. Works in bull/bear by filtering with 1d EMA50 trend.
Target: 20-40 trades/year on 4h.
"""

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
    
    # Get 1d data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 4h Donchian channels (20-period)
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    for i in range(20, n):
        high_max[i] = np.max(high[i-20:i])
        low_min[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.5 x 30-period average
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high AND above 1d EMA50, with volume
            if (close[i] > high_max[i] and 
                close[i] > ema_50_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND below 1d EMA50, with volume
            elif (close[i] < low_min[i] and 
                  close[i] < ema_50_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to Donchian low or breaks below 1d EMA50
            if (not np.isnan(low_min[i]) and close[i] < low_min[i]) or \
               (not np.isnan(ema_50_aligned[i]) and close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian high or breaks above 1d EMA50
            if (not np.isnan(high_max[i]) and close[i] > high_max[i]) or \
               (not np.isnan(ema_50_aligned[i]) and close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0