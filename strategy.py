#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for trend filter and price channel
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Donchian channels (20-period)
    highest_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_12h)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # 1d EMA 50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 12h volume for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(donchian_high_12h_aligned[i]) or np.isnan(donchian_low_12h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        
        if position == 0:
            # Long: breakout above 12h Donchian high + above daily EMA50 + volume confirmation
            if (price > donchian_high_12h_aligned[i] and 
                price > ema_50_1d_aligned[i] and 
                vol > 1.5 * vol_ma_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below 12h Donchian low + below daily EMA50 + volume confirmation
            elif (price < donchian_low_12h_aligned[i] and 
                  price < ema_50_1d_aligned[i] and 
                  vol > 1.5 * vol_ma_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 12h Donchian low or below daily EMA50
            if price < donchian_low_12h_aligned[i] or price < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 12h Donchian high or above daily EMA50
            if price > donchian_high_12h_aligned[i] or price > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0