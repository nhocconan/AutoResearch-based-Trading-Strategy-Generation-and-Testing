#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for trend filter
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Donchian channels (20-period) on 1d data
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 4h timeframe
    ema_200_4h = align_htf_to_ltf(prices, df_1d, ema_200)
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: volume > 1.8x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_200_4h[i]) or 
            np.isnan(donchian_high_4h[i]) or 
            np.isnan(donchian_low_4h[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < Donchian low or trend fails
            if close[i] < donchian_low_4h[i] or close[i] < ema_200_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high or trend fails
            if close[i] > donchian_high_4h[i] or close[i] > ema_200_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price > Donchian high + uptrend + volume
            if (close[i] > donchian_high_4h[i] and 
                close[i] > ema_200_4h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price < Donchian low + downtrend + volume
            elif (close[i] < donchian_low_4h[i] and 
                  close[i] < ema_200_4h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals