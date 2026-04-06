#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14069_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_ema(arr, period):
    return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA200 on 1d
    ema200_1d = calculate_ema(close_1d, 200)
    
    # Align EMA200 to 4h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    highest_20 = np.full_like(high, np.nan)
    lowest_20 = np.full_like(low, np.nan)
    for i in range(19, n):
        highest_20[i] = np.max(high[i-19:i+1])
        lowest_20[i] = np.min(low[i-19:i+1])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(19, 200) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or \
           np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume filter: current volume > 1.5x average volume
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Generate signals
        if position == 0:
            # Long: price breaks above Donchian upper + above EMA200 + volume filter
            if close[i] > highest_20[i] and close[i] > ema200_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short: price breaks below Donchian lower + below EMA200 + volume filter
            elif close[i] < lowest_20[i] and close[i] < ema200_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or price breaks below Donchian lower
            if close[i] <= stop_price or close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or price breaks above Donchian upper
            if close[i] >= stop_price or close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals