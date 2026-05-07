#!/usr/bin/env python3
name = "1d_WeeklyDonchian20_Trend_Volume"
timeframe = "1d"
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
    
    # Get 1w data for trend filter (EMA30) and Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA30 for trend filter
    close_1w = df_1w['close'].values
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # Calculate Donchian channels from previous 1w candle (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Shift to get previous 1w period's values for Donchian calculation
    high_1w_shifted = np.roll(high_1w, 1)
    low_1w_shifted = np.roll(low_1w, 1)
    
    # Calculate 20-period rolling max and min from shifted arrays
    # We need to use pandas rolling on the shifted arrays
    high_20 = pd.Series(high_1w_shifted).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w_shifted).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Calculate volume confirmation (current volume vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_30_1w_aligned[i]) or 
            np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian, uptrend (price > EMA30), volume confirmation
            if (close[i] > high_20_aligned[i] and 
                close[i] > ema_30_1w_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, downtrend (price < EMA30), volume confirmation
            elif (close[i] < low_20_aligned[i] and 
                  close[i] < ema_30_1w_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian (reversal signal)
            if close[i] < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian (reversal signal)
            if close[i] > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals