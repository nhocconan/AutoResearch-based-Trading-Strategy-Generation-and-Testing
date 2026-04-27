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
    
    # Get 12h data for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Donchian breakout
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian(20) on daily: upper = max(high_1d[last 20]), lower = min(low_1d[last 20])
    donch_upper_1d = np.full(len(high_1d), np.nan)
    donch_lower_1d = np.full(len(low_1d), np.nan)
    for i in range(19, len(high_1d)):
        donch_upper_1d[i] = np.max(high_1d[i-19:i+1])
        donch_lower_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian to 4h
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper_1d)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower_1d)
    
    # Volume filter: volume > 1.5 x 20-period average (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), EMA (50), volume MA (20)
    start_idx = max(19, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter: 12h EMA(50)
        bullish_trend = price > ema_50_12h_aligned[i]
        bearish_trend = price < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: break above Donchian upper with volume and trend
            if price > donch_upper_aligned[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: break below Donchian lower with volume and trend
            elif price < donch_lower_aligned[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price re-enters Donchian channel (below upper) or trend fails
            if price < donch_upper_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters Donchian channel (above lower) or trend fails
            if price > donch_lower_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0