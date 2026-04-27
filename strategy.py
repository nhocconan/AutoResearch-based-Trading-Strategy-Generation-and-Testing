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
    
    # Get 12h data for price structure (Donchian channel)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channel (20-period) on 12h data
    donchian_high = np.full(len(df_12h), np.nan)
    donchian_low = np.full(len(df_12h), np.nan)
    for i in range(19, len(df_12h)):
        donchian_high[i] = np.max(high_12h[i-19:i+1])
        donchian_low[i] = np.min(low_12h[i-19:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1d data for volume filter (volume > 2.0x 20-period average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = np.full(len(df_1d), np.nan, dtype=np.float64)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get 1d data for trend filter (EMA50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h Donchian (20), 1d volume MA (20), 1d EMA (50)
    start_idx = max(20, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20_1d_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter: price above/below EMA50
        bullish_trend = price > ema_trend
        bearish_trend = price < ema_trend
        
        if position == 0:
            # Long: price breaks above Donchian high + bullish trend + volume spike
            if price > donch_high and bullish_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + bearish trend + volume spike
            elif price < donch_low and bearish_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low or trend turns bearish
            if price <= donch_low or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian high or trend turns bullish
            if price >= donch_high or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0