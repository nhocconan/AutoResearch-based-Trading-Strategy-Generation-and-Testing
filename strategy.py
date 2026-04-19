#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d trend filter (EMA 50)
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h timeframe
    high_20_4h = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_4h = align_htf_to_ltf(prices, df_1d, low_20)
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40
    
    for i in range(start_idx, n):
        if np.isnan(high_20_4h[i]) or np.isnan(low_20_4h[i]) or np.isnan(ema_50_4h[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above 1d Donchian high + above 1d EMA50 + volume spike
            if price > high_20_4h[i] and price > ema_50_4h[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1d Donchian low + below 1d EMA50 + volume spike
            elif price < low_20_4h[i] and price < ema_50_4h[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below 1d Donchian low (reversal signal)
            if price < low_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above 1d Donchian high (reversal signal)
            if price > high_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals