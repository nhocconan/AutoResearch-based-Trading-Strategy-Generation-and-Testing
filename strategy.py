#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA trend filter (50-period)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) from daily data (previous 20 days)
    high_max_20 = np.full(len(high_1d), np.nan)
    low_min_20 = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        high_max_20[i] = np.max(high_1d[i-20:i])
        low_min_20[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 6h timeframe
    high_max_20_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    
    # Volume filter: volume > 2.0 x 24-period average (4h periods = 4 days)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly EMA (50), Donchian (20), volume MA (24)
    start_idx = max(50, 20, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_max_20_aligned[i]) or
            np.isnan(low_min_20_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter from weekly EMA
        bullish_trend = price > ema_50_1w_aligned[i]
        bearish_trend = price < ema_50_1w_aligned[i]
        
        upper = high_max_20_aligned[i]
        lower = low_min_20_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume + bullish weekly trend
            if price > upper and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian + volume + bearish weekly trend
            elif price < lower and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian or trend turns bearish
            if price < lower or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above upper Donchian or trend turns bullish
            if price > upper or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_WeeklyTrend_Volume_Breakout"
timeframe = "6h"
leverage = 1.0