#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data once before loop (for 12h timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate daily ATR(14) for volatility filter
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - daily_close_prev),
                               np.abs(daily_low - daily_close_prev)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_30 = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    volatility_ratio = atr_14 / (atr_ma_30 + 1e-10)
    
    # Calculate daily Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = daily_volume / (vol_ma_20 + 1e-10)
    
    # Align HTF indicators to 12h timeframe with proper delay
    volatility_ratio_12h = align_htf_to_ltf(prices, df_1d, volatility_ratio)
    highest_20_12h = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_12h = align_htf_to_ltf(prices, df_1d, lowest_20)
    volume_ratio_12h = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(volatility_ratio_12h[i]) or 
            np.isnan(highest_20_12h[i]) or 
            np.isnan(lowest_20_12h[i]) or 
            np.isnan(volume_ratio_12h[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Volatility regime: only trade in normal/high volatility (avoid low vol squeezes)
        # 2. Daily Donchian breakout with volume confirmation
        # 3. Discrete position sizing: 0.25
        
        # Long conditions
        if (volatility_ratio_12h[i] > 0.8 and      # Avoid low volatility squeezes
            close[i] > highest_20_12h[i] and       # Daily Donchian breakout
            volume_ratio_12h[i] > 1.5):            # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions
        elif (volatility_ratio_12h[i] > 0.8 and    # Avoid low volatility squeezes
              close[i] < lowest_20_12h[i] and      # Daily Donchian breakdown
              volume_ratio_12h[i] > 1.5):          # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_DailyVolatility_Donchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0