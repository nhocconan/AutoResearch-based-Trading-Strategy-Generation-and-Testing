#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # 1d data for Donchian channels and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 1d
    n1 = len(high_1d)
    upper_20 = np.full(n1, np.nan)
    lower_20 = np.full(n1, np.nan)
    
    for i in range(19, n1):
        upper_20[i] = np.max(high_1d[i-19:i+1])
        lower_20[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 50-period EMA for 1d trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 4h timeframe
    upper_20_4h = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_4h = align_htf_to_ltf(prices, df_1d, lower_20)
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 4h volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian or trend fails
            if close[i] < lower_20_4h[i] or close[i] < ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian or trend fails
            if close[i] > upper_20_4h[i] or close[i] > ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_50_4h[i]
            bearish = close[i] < ema_50_4h[i]
            
            # Long: price breaks above upper Donchian + bullish trend + volume
            if (close[i] > upper_20_4h[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower Donchian + bearish trend + volume
            elif (close[i] < lower_20_4h[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals