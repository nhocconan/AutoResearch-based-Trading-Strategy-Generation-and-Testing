#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h volume confirmation and 1d trend filter
# Donchian(20) captures breakouts in trending markets
# 12h volume > 1.5x 24-period average filters weak breakouts
# 1d EMA50 provides higher timeframe bias to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "6h_Donchian20_12hVolume_1dEMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 24:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=24, min_periods=24).mean().values
    vol_ratio = vol_12h / vol_ma_12h
    vol_confirm_12h = align_htf_to_ltf(prices, df_12h, vol_ratio > 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 24, 50)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_confirm_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + above 1d EMA50 + volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + below 1d EMA50 + volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower Donchian or below 1d EMA50
            if (close[i] < lowest_low[i]) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper Donchian or above 1d EMA50
            if (close[i] > highest_high[i]) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals