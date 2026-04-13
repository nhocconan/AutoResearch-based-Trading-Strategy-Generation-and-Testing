#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d average volume (20-period)
    vol_series = pd.Series(df_1d['volume'].values)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA200 trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    avg_vol_aligned = align_htf_to_ltf(prices, df_1d, avg_vol)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 200)
    for i in range(start, n):
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(avg_vol_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: breakout above upper band + volume confirmation + price above EMA200
            if (price > upper_aligned[i] and vol > 2.0 * avg_vol_aligned[i] and price > ema_200_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: breakout below lower band + volume confirmation + price below EMA200
            elif (price < lower_aligned[i] and vol > 2.0 * avg_vol_aligned[i] and price < ema_200_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower band OR below EMA200
            if price < lower_aligned[i] or price < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper band OR above EMA200
            if price > upper_aligned[i] or price > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Volume_EMA200Trend_v2"
timeframe = "4h"
leverage = 1.0