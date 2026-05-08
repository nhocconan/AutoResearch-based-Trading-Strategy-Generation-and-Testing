#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Donchian20_1wTrend_Volume_breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(30) on weekly close for trend filter
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # Daily Donchian channels (20-period)
    # Using pandas rolling on daily high/low from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe (wait for daily close)
    donchian_high_d = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_d = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: 20-period average on daily
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_d[i]) or np.isnan(donchian_low_d[i]) or 
            np.isnan(ema_30_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + above weekly EMA30 + volume
            if (close[i] > donchian_high_d[i] and
                close[i] > ema_30_1w_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below weekly EMA30 + volume
            elif (close[i] < donchian_low_d[i] and
                  close[i] < ema_30_1w_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price falls back below Donchian low or below weekly EMA30
            if (close[i] < donchian_low_d[i] or
                close[i] < ema_30_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price rises back above Donchian high or above weekly EMA30
            if (close[i] > donchian_high_d[i] or
                close[i] > ema_30_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals