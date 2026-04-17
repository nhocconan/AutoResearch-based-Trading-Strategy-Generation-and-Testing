#!/usr/bin/env python3
"""
4h_Donchian_Breakout_VolumeTrend_v1
Breakout above/below Donchian(20) with volume confirmation (volume > 1.5x SMA30 volume)
and EMA50 trend filter. Exit when price returns to middle of Donchian channel.
Designed for 4h timeframe with 1d HTF trend alignment.
Target: 50-150 total trades over 4 years (12-37/year).
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
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_high + lowest_low) / 2.0
    
    # === Volume confirmation: volume > 1.5x SMA30 volume ===
    vol_sma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_threshold = 1.5 * vol_sma30
    
    # === EMA50 for trend filter (1d HTF) ===
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
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(vol_sma30[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper Donchian, volume confirmation, price above 1d EMA50
            if (close[i] > highest_high[i] and 
                volume[i] > vol_threshold[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian, volume confirmation, price below 1d EMA50
            elif (close[i] < lowest_low[i] and 
                  volume[i] > vol_threshold[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to middle of Donchian channel
        elif position == 1:
            # Exit long: price crosses below Donchian middle
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian middle
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0