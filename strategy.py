#!/usr/bin/env python3
name = "6h_Donchian_20_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # Daily ATR for stop loss and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 00-08 UTC (Asian session breakout)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 0) & (hours < 8)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if np.isnan(sma_20_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above upper band, above weekly SMA, volume spike, Asian session
            if (close[i] > high_max_20[i] and 
                close[i] > sma_20_1w_aligned[i] and 
                volume[i] > 1.5 * volume_ma20[i] and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band, below weekly SMA, volume spike, Asian session
            elif (close[i] < low_min_20[i] and 
                  close[i] < sma_20_1w_aligned[i] and 
                  volume[i] > 1.5 * volume_ma20[i] and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below lower band or below weekly SMA
            if close[i] < low_min_20[i] or close[i] < sma_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above upper band or above weekly SMA
            if close[i] > high_max_20[i] or close[i] > sma_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals