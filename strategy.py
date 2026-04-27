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
    
    # Get 6h data for EMA200 trend filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    # 6h EMA200 for trend filter
    close_6h = df_6h['close'].values
    ema_200_6h = pd.Series(close_6h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_200_6h)
    
    # Get 1d data for Donchian(20) breakout
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Donchian(20) channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: volume > 1.8x 30-period average
    vol_ma_30 = np.full(n, np.nan, dtype=np.float64)
    for i in range(29, n):
        vol_ma_30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 6h EMA (200 periods), daily Donchian (20 periods), volume MA (30 periods)
    start_idx = max(200, 20, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_6h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        ema_trend = ema_200_6h_aligned[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_30[i]
        
        # Volume filter: volume > 1.8x average
        vol_filter = vol_now > 1.8 * vol_avg
        
        if position == 0:
            # Long: price breaks above daily Donchian high + above 6h EMA200 + volume spike
            if price > upper_band and price > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below daily Donchian low + below 6h EMA200 + volume spike
            elif price < lower_band and price < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 6h EMA200 (trend following)
            if price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to 6h EMA200 (trend following)
            if price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_20_EMA200_Trend_Volume"
timeframe = "6h"
leverage = 1.0