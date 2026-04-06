#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14056_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA and volume average (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA(50) and volume average
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # 12h data for Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 50 for EMA, 20 for Donchian, 14 for ATR)
    start = max(50, 20, 14)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or \
           np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(atr[i]):
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
        
        # Volume confirmation: current volume > 1.5 * 20-day average volume
        vol_confirm = volume[i] > 1.5 * vol_avg_20_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long: price breaks above Donchian high + above EMA + volume confirmation
            if high[i] > high_20[i-1] and close[i] > ema_50_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short: price breaks below Donchian low + below EMA + volume confirmation
            elif low[i] < low_20[i-1] and close[i] < ema_50_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or price breaks below Donchian low
            if close[i] <= stop_price or low[i] < low_20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or price breaks above Donchian high
            if close[i] >= stop_price or high[i] > high_20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals