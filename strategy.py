#!/usr/bin/env python3
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
    
    # Get daily data for Supertrend (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Supertrend (ATR-based)
    period = 10
    multiplier = 3.0
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    hl2 = (high_1d + low_1d) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros_like(close_1d)
    dir_ = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    dir_[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            dir_[i] = 1
        else:
            dir_[i] = -1
        
        if dir_[i] == 1:
            supertrend[i] = max(lower[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper[i], supertrend[i-1])
    
    # Align daily Supertrend direction to 4h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1d, dir_)
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need daily Supertrend, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(supertrend_dir_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        if position == 0:
            # Long: price breaks above Donchian high with daily uptrend and volume
            if (close[i] > highest_high[i] and supertrend_dir_aligned[i] == 1 and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with daily downtrend and volume
            elif (close[i] < lowest_low[i] and supertrend_dir_aligned[i] == -1 and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian low
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian high
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Supertrend_DonchianBreakout_Volume"
timeframe = "4h"
leverage = 1.0