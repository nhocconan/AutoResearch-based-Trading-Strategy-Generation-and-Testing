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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h close for trend filter
    close_4h = df_4h['close'].values
    # 1d Williams %R for overbought/oversold
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Calculate 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe with proper delay
    williams_r_1h = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    ema_50_1h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Session filter: 08-20 UTC (pre-compute hours for efficiency)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_1h[i]) or np.isnan(ema_50_1h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1d Williams %R extreme (oversold/overbought)
        # 2. 1h Donchian breakout in direction of Williams %R signal
        # 3. 4h trend filter: price above/below 4h EMA50
        # 4. 1h volume confirmation: volume > 1.5x average
        
        # Long conditions: Williams %R oversold (< -80) + Donchian breakout above in uptrend
        if (williams_r_1h[i] < -80 and          # Daily oversold
            close[i] > ema_50_1h[i] and         # 4h uptrend filter
            close[i] > highest_20[i] and        # 1h Donchian breakout
            volume_ratio[i] > 1.5):             # Volume confirmation
            signals[i] = 0.20
            
        # Short conditions: Williams %R overbought (> -20) + Donchian breakdown below in downtrend
        elif (williams_r_1h[i] > -20 and        # Daily overbought
              close[i] < ema_50_1h[i] and       # 4h downtrend filter
              close[i] < lowest_20[i] and       # 1h Donchian breakdown
              volume_ratio[i] > 1.5):           # Volume confirmation
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_WilliamsR_Donchian_Breakout_EMA50_Volume_SessionFilter"
timeframe = "1h"
leverage = 1.0