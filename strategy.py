#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_price_channel_4h_1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend and price channel
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 4h price channel (Donchian-like: 20-period high/low)
    high_max_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    mid_4h = (high_max_4h + low_min_4h) / 2.0
    
    # Align 4h price channel to 1h
    high_max_4h_1h = align_htf_to_ltf(prices, df_4h, high_max_4h)
    low_min_4h_1h = align_htf_to_ltf(prices, df_4h, low_min_4h)
    mid_4h_1h = align_htf_to_ltf(prices, df_4h, mid_4h)
    
    # 1d trend: 50-period EMA
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_1h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 1h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max_4h_1h[i]) or np.isnan(low_min_4h_1h[i]) or 
            np.isnan(mid_4h_1h[i]) or np.isnan(ema_50_1d_1h[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below 4h channel mid or 1d trend fails
            if close[i] < mid_4h_1h[i] or close[i] < ema_50_1d_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price above 4h channel mid or 1d trend fails
            if close[i] > mid_4h_1h[i] or close[i] > ema_50_1d_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: price above 4h channel high + above 1d EMA + volume
            if (close[i] > high_max_4h_1h[i] and 
                close[i] > ema_50_1d_1h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.20
            # Short: price below 4h channel low + below 1d EMA + volume
            elif (close[i] < low_min_4h_1h[i] and 
                  close[i] < ema_50_1d_1h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.20
    
    return signals