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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Williams %R (14-period) for mean reversion signals
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values: 0 to -100, where > -20 is overbought, < -80 is oversold
    hh_14 = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (hh_14 - df_12h['close'].values) / (hh_14 - ll_14 + 1e-10) * -100
    
    # Calculate 12h ATR (14-period) for volatility filter
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_prev = np.concatenate([[close_12h[0]], close_12h[:-1]])
    tr = np.maximum(high_12h - low_12h,
                    np.maximum(np.abs(high_12h - close_prev),
                               np.abs(low_12h - close_prev)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_14 > (atr_ma_50 * 0.5)  # Trade only when ATR > 50% of its MA
    
    # Align HTF indicators to 6h timeframe with proper delay
    williams_r_6h = align_htf_to_ltf(prices, df_12h, williams_r)
    volatility_filter_6h = align_htf_to_ltf(prices, df_12h, volatility_filter)
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(volatility_filter_6h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Williams %R extreme readings for mean reversion (oversold/overbought)
        # 2. Volatility filter: only trade when volatility is expanding
        # 3. Donchian breakout in direction of mean reversion
        # 4. Volume confirmation
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: oversold + volume + breakout
        if (williams_r_6h[i] < -80 and      # Oversold condition
            volatility_filter_6h[i] and     # Volatility expanding
            close[i] > highest_20[i] and    # Donchian breakout
            volume_ratio[i] > 1.3):         # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: overbought + volume + breakdown
        elif (williams_r_6h[i] > -20 and   # Overbought condition
              volatility_filter_6h[i] and  # Volatility expanding
              close[i] < lowest_20[i] and  # Donchian breakdown
              volume_ratio[i] > 1.3):      # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_Donchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0