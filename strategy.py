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
    highest_20d = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) for volatility filter
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - daily_close_prev),
                               np.abs(daily_low - daily_close_prev)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    volatility_ratio = atr_14 / (atr_ma_50 + 1e-10)
    
    # Align HTF indicators to 4h timeframe with proper delay
    highest_20d_4h = align_htf_to_ltf(prices, df_1d, highest_20d)
    lowest_20d_4h = align_htf_to_ltf(prices, df_1d, lowest_20d)
    volatility_ratio_4h = align_htf_to_ltf(prices, df_1d, volatility_ratio)
    
    # Calculate 4h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20d_4h[i]) or np.isnan(lowest_20d_4h[i]) or 
            np.isnan(volatility_ratio_4h[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Daily Donchian breakout/breakdown
        # 2. Volatility regime: only trade in normal/high volatility (avoid low vol squeezes)
        # 3. 4h price must confirm the daily breakout direction
        # 4. Volume confirmation on 4h chart
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: price breaks above daily Donchian upper band
        if (close[i] > highest_20d_4h[i] and     # Daily Donchian breakout
            volatility_ratio_4h[i] > 0.8 and     # Avoid low volatility squeezes
            close[i] > highest_20[i] and         # 4h confirms upward momentum
            volume_ratio[i] > 1.5):              # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: price breaks below daily Donchian lower band
        elif (close[i] < lowest_20d_4h[i] and    # Daily Donchian breakdown
              volatility_ratio_4h[i] > 0.8 and   # Avoid low volatility squeezes
              close[i] < lowest_20[i] and        # 4h confirms downward momentum
              volume_ratio[i] > 1.5):            # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyDonchian_Breakout_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0