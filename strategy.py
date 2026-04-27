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
    
    # Get 1d data for Donchian channel and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Donchian channel (20-period)
    # Upper = max(high_1d over last 20 days)
    # Lower = min(low_1d over last 20 days)
    donchian_upper = np.full(len(high_1d), np.nan)
    donchian_lower = np.full(len(low_1d), np.nan)
    for i in range(19, len(high_1d)):
        donchian_upper[i] = np.max(high_1d[i-19:i+1])
        donchian_lower[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian levels to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # 1d EMA (50-period) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.5 x 48-period average (6h bars = 2 per day, so 48 = ~24 days)
    vol_ma_48 = np.full(n, np.nan)
    for i in range(47, n):
        vol_ma_48[i] = np.mean(volume[i-47:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20 days), EMA (50), volume MA (48)
    start_idx = max(20, 50, 48)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_48[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume + bullish 1d trend
            if price > upper and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian lower + volume + bearish 1d trend
            elif price < lower and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian lower or trend turns bearish
            if price < lower or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian upper or trend turns bullish
            if price > upper or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_20_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0