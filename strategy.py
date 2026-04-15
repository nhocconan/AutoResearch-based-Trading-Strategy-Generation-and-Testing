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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate 20-period daily Donchian channels
    highest_20 = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period daily EMA for trend filter
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    highest_20_6h = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_6h = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Calculate 6h Donchian channels (20-period)
    highest_20_6h_local = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20_6h_local = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_6h[i]) or np.isnan(highest_20_6h[i]) or 
            np.isnan(lowest_20_6h[i]) or np.isnan(highest_20_6h_local[i]) or 
            np.isnan(lowest_20_6h_local[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Daily trend filter: price above/below daily EMA50
        # 2. 6h Donchian breakout with volume confirmation
        # 3. Only trade when 6h price breaks beyond daily Donchian levels (strong breakout)
        # 4. Discrete position sizing: 0.25
        
        # Long conditions
        if (close[i] > ema_50_6h[i] and  # Uptrend filter
            close[i] > highest_20_6h_local[i] and     # 6h Donchian breakout
            close[i] > highest_20_6h[i] and        # Break above daily resistance (strong breakout)
            volume_ratio[i] > 1.5):                # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < ema_50_6h[i] and   # Downtrend filter
              close[i] < lowest_20_6h_local[i] and      # 6h Donchian breakdown
              close[i] < lowest_20_6h[i] and       # Break below daily support (strong breakdown)
              volume_ratio[i] > 1.5):              # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_DailyEMA_Donchian_Breakout_Strong"
timeframe = "6h"
leverage = 1.0