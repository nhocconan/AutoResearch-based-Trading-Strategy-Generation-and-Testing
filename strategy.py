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
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily ATR ratio (current ATR / 50-period ATR) to detect volatility expansion
    atr_50 = pd.Series(atr_14).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr_14 / (atr_50 + 1e-10)
    
    # Align HTF indicators to 6h timeframe with proper delay
    atr_ratio_6h = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_6h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Volatility expansion: ATR ratio > 1.2 (current ATR > 120% of 50-day ATR)
        # 2. 6h price breaks above 20-period high with volume confirmation → long
        # 3. 6h price breaks below 20-period low with volume confirmation → short
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.25
        
        # Volatility expansion filter
        vol_expansion = atr_ratio_6h[i] > 1.2
        
        # Long conditions: 6h breakout above Donchian high
        if (close[i] > highest_20[i] and            # 6h price above Donchian high
            volume_ratio[i] > 1.5 and               # Volume confirmation
            vol_expansion):                         # Volatility expansion
            signals[i] = 0.25
            
        # Short conditions: 6h breakdown below Donchian low
        elif (close[i] < lowest_20[i] and           # 6h price below Donchian low
              volume_ratio[i] > 1.5 and             # Volume confirmation
              vol_expansion):                       # Volatility expansion
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_VolExpansion_Donchian_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0