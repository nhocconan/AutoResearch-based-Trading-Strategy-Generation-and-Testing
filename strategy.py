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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA20 for trend
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate 12h ATR14 for volatility filter
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_prev = np.concatenate([[close_12h[0]], close_12h[:-1]])
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - close_12h_prev),
                                   np.abs(low_12h - close_12h_prev)))
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_ma_50_12h = pd.Series(atr_14_12h).rolling(window=50, min_periods=50).mean().values
    atr_ratio_12h = atr_14_12h / (atr_ma_50_12h + 1e-10)
    atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h)
    
    # Calculate 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(atr_ratio_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 12h trend filter: price above/below 12h EMA20
        # 2. Volatility regime: only trade when ATR ratio > 0.8 (avoid low volatility squeezes)
        # 3. 6h Donchian breakout with volume confirmation (>1.5x average volume)
        # 4. Discrete position sizing: 0.25
        
        # Long conditions
        if (close[i] > ema_20_12h_aligned[i] and  # 12h uptrend filter
            atr_ratio_12h_aligned[i] > 0.8 and    # Avoid low volatility squeezes
            close[i] > highest_20[i] and          # 6h Donchian breakout
            volume_ratio[i] > 1.5):               # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < ema_20_12h_aligned[i] and   # 12h downtrend filter
              atr_ratio_12h_aligned[i] > 0.8 and     # Avoid low volatility squeezes
              close[i] < lowest_20[i] and            # 6h Donchian breakdown
              volume_ratio[i] > 1.5):                # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_12hEMA20_ATR_Volume_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0