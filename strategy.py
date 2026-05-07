#!/usr/bin/env python3
name = "1h_4h1d_Donchian20_Trend_Volume"
timeframe = "1h"
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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1h volume spike: > 2.0x 20-period average
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_1h = volume > 2.0 * vol_ma_1h
    
    # Session filter: 08-20 UTC (already datetime64[ms])
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 4h Donchian high + volume spike + above 1d EMA34
            if (close[i] > donchian_high_4h_aligned[i] and vol_spike_1h[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Break below 4h Donchian low + volume spike + below 1d EMA34
            elif (close[i] < donchian_low_4h_aligned[i] and vol_spike_1h[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Break below 4h Donchian low or below 1d EMA34
            if close[i] < donchian_low_4h_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Break above 4h Donchian high or above 1d EMA34
            if close[i] > donchian_high_4h_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals