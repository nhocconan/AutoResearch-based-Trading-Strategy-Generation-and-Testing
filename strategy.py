#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA21 trend filter and volume confirmation
# Donchian breakouts capture momentum in trending markets; EMA21 filters counter-trend trades
# Volume confirmation ensures breakouts have conviction. Designed for low trade frequency
# and robustness in both bull and bear regimes. Target: 50-150 total trades over 4 years.
name = "12h_Donchian20_1dEMA21_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA21 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    ema_21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Donchian channels (20-period) on 12h
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(ema_21_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian + above 1d EMA21 + volume confirmation
            if (close[i] > highest[i] and 
                close[i] > ema_21_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian + below 1d EMA21 + volume confirmation
            elif (close[i] < lowest[i] and 
                  close[i] < ema_21_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower Donchian or below 1d EMA21
            if (close[i] < lowest[i]) or (close[i] < ema_21_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper Donchian or above 1d EMA21
            if (close[i] > highest[i]) or (close[i] > ema_21_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals