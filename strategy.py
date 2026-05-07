#!/usr/bin/env python3
name = "4h_Donchian20_Turnaround_Volume"
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
    
    # Get 1d data for trend filter (EMA50) and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) from previous 1d candle (upper and lower)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift to get previous 1d period's values
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    
    # Calculate Donchian levels: highest high and lowest low of previous 20 periods
    high_20 = np.zeros(len(high_1d_shifted))
    low_20 = np.zeros(len(low_1d_shifted))
    for i in range(len(high_1d_shifted)):
        start_idx = max(0, i - 19)
        high_20[i] = np.max(high_1d_shifted[start_idx:i+1])
        low_20[i] = np.min(low_1d_shifted[start_idx:i+1])
    
    # Align Donchian levels to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate volume confirmation (current volume vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, uptrend (price > EMA50), volume confirmation
            if (close[i] > high_20_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, downtrend (price < EMA50), volume confirmation
            elif (close[i] < low_20_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower (reversal signal)
            if close[i] < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper (reversal signal)
            if close[i] > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals