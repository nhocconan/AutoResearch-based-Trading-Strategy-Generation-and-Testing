#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14065_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_donchian(high, low, window):
    """Calculate Donchian Channels: upper and lower bands"""
    upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
    return upper, lower

def calculate_ema(values, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_volume_ma(volume, period):
    """Calculate Volume Moving Average"""
    return pd.Series(volume).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian, EMA, and volume (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d indicators
    donchian_upper_1d, donchian_lower_1d = calculate_donchian(high_1d, low_1d, 20)
    ema_1d = calculate_ema(close_1d, 50)
    volume_ma_1d = calculate_volume_ma(volume_1d, 20)
    
    # Align 1d indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # 12h data for entry timing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 50 for EMA)
    start = max(20, 50)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or \
           np.isnan(ema_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(volume[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check for exit conditions (price crosses EMA)
        if position == 1:  # long position
            if close[i] <= ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:  # flat position
            # Long signal: price breaks above Donchian upper + volume confirmation
            if high[i] > donchian_upper_aligned[i] and volume[i] > volume_ma_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short signal: price breaks below Donchian lower + volume confirmation
            elif low[i] < donchian_lower_aligned[i] and volume[i] > volume_ma_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals