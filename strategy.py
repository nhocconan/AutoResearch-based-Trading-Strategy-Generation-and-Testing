#!/usr/bin/env python3
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
    
    # Load 1d data for daily close and weekly for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA (50-period) for trend
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 1w data for weekly trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA (20-period) for long-term trend
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 + ema_20_1w[i-1] * 18) / 20
    
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily volume average (20-period)
    vol_ma_20_1d = np.full_like(df_1d['volume'].values, np.nan)
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        for i in range(19, len(vol_1d)):
            vol_ma_20_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-period average
        if vol_ma_20_1d_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: Close > 1d EMA50 AND weekly EMA20 rising AND volume surge
            weekly_rising = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1] if i > 0 else False
            if (close[i] > ema_50_1d_aligned[i] and 
                weekly_rising and 
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Close < 1d EMA50 AND weekly EMA20 falling AND volume surge
            elif (close[i] < ema_50_1d_aligned[i] and 
                  not weekly_rising and 
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Close < 1d EMA50 OR volume drops
            if (close[i] < ema_50_1d_aligned[i] or 
                volume_ratio < 1.0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Close > 1d EMA50 OR volume drops
            if (close[i] > ema_50_1d_aligned[i] or 
                volume_ratio < 1.0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_EMA50_EMA20_Volume"
timeframe = "1d"
leverage = 1.0