#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop (required for MTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators on daily data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Donchian channels (20-period) - using previous day's values
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    upper_1d = high_series_1d.rolling(window=20, min_periods=20).max().shift(1).values
    lower_1d = low_series_1d.rolling(window=20, min_periods=20).min().shift(1).values
    
    # 1d average volume (20-period) - previous day
    vol_series_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_series_1d.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # 1d EMA200 trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 1d indicators to 4h timeframe
    upper_4h = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_4h = align_htf_to_ltf(prices, df_1d, lower_1d)
    avg_vol_4h = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    ema_200_4h = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 200)  # Ensure enough data for indicators
    for i in range(start, n):
        if (np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]) or 
            np.isnan(avg_vol_4h[i]) or np.isnan(ema_200_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: breakout above upper band + volume confirmation + price above EMA200
            if (price > upper_4h[i] and vol > 2.0 * avg_vol_4h[i] and price > ema_200_4h[i]):
                position = 1
                signals[i] = position_size
            # Short: breakout below lower band + volume confirmation + price below EMA200
            elif (price < lower_4h[i] and vol > 2.0 * avg_vol_4h[i] and price < ema_200_4h[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower band OR below EMA200
            if price < lower_4h[i] or price < ema_200_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper band OR above EMA200
            if price > upper_4h[i] or price > ema_200_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Volume_EMA200Trend"
timeframe = "4h"
leverage = 1.0