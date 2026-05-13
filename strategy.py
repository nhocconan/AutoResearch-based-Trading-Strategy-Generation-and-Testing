#!/usr/bin/env python3
name = "6h_PriceChannel_Breakout_Volume_Trend"
timeframe = "6h"
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
    
    # Load 1D data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA20 on 1D for trend filter
    close_1d_series = pd.Series(close_1d)
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1D EMA20 to 6H timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate 6H Donchian channel (20-period)
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    
    for i in range(n):
        start_idx = max(0, i - 19)
        if i >= 19:
            high_max[i] = np.max(high[start_idx:i+1])
            low_min[i] = np.min(low[start_idx:i+1])
    
    # Calculate volume ratio (current vs 20-period average)
    volume_ma = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    volume_ratio = np.full(n, np.nan)
    volume_ratio = volume / np.where(volume_ma == 0, np.nan, volume_ma)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema20_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1D EMA20
        price_above_1d_ema = close[i] > ema20_1d_aligned[i]
        price_below_1d_ema = close[i] < ema20_1d_aligned[i]
        
        # Volume filter: above average volume
        volume_filter = volume_ratio[i] > 1.5
        
        # Long signal: break above Donchian high + uptrend + volume
        if (close[i] > high_max[i] and price_above_1d_ema and volume_filter):
            signals[i] = 0.25
        # Short signal: break below Donchian low + downtrend + volume
        elif (close[i] < low_min[i] and price_below_1d_ema and volume_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals