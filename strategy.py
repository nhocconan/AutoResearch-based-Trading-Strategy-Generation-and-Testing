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
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_volume = df_1w['volume'].values
    
    # Calculate 50-period weekly EMA for trend filter
    ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period weekly ATR for volatility regime
    weekly_close_prev = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    tr = np.maximum(weekly_high - weekly_low,
                    np.maximum(np.abs(weekly_high - weekly_close_prev),
                               np.abs(weekly_low - weekly_close_prev)))
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ma_50 = pd.Series(atr_20).rolling(window=50, min_periods=50).mean().values
    volatility_ratio = atr_20 / (atr_ma_50 + 1e-10)
    
    # Calculate weekly volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(weekly_volume).rolling(window=20, min_periods=20).mean().values
    weekly_volume_ratio = weekly_volume / (vol_ma_20 + 1e-10)
    
    # Align HTF indicators to 6h timeframe with proper delay
    ema_50_6h = align_htf_to_ltf(prices, df_1w, ema_50)
    volatility_ratio_6h = align_htf_to_ltf(prices, df_1w, volatility_ratio)
    weekly_volume_ratio_6h = align_htf_to_ltf(prices, df_1w, weekly_volume_ratio)
    
    # Calculate 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio_6h = volume / (vol_ma_20_6h + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_6h[i]) or np.isnan(volatility_ratio_6h[i]) or 
            np.isnan(weekly_volume_ratio_6h[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio_6h[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above/below weekly EMA50
        # 2. Volatility regime: only trade in normal/high volatility (avoid low vol squeezes)
        # 3. Weekly volume confirmation: above average volume on weekly timeframe
        # 4. 6h Donchian breakout with volume confirmation
        # 5. Discrete position sizing: 0.25
        
        # Long conditions
        if (close[i] > ema_50_6h[i] and  # Uptrend filter
            volatility_ratio_6h[i] > 0.8 and  # Avoid low volatility squeezes
            weekly_volume_ratio_6h[i] > 1.2 and  # Weekly volume confirmation
            close[i] > highest_20[i] and     # 6h Donchian breakout
            volume_ratio_6h[i] > 1.5):       # 6h volume confirmation
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < ema_50_6h[i] and   # Downtrend filter
              volatility_ratio_6h[i] > 0.8 and  # Avoid low volatility squeezes
              weekly_volume_ratio_6h[i] > 1.2 and  # Weekly volume confirmation
              close[i] < lowest_20[i] and      # 6h Donchian breakdown
              volume_ratio_6h[i] > 1.5):       # 6h volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA_Volume_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0