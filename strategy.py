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
    
    # Calculate daily ATR(14) for volatility regime
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr_daily = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily_14 = pd.Series(tr_daily).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily ATR ratio: current ATR / 50-day ATR average (volatility regime filter)
    atr_ma_50 = pd.Series(atr_daily_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_daily_14 / (atr_ma_50 + 1e-10)
    
    # Align HTF ATR ratio to 6h timeframe
    atr_ratio_6h = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_ratio_6h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 6h price breaks above 20-period Donchian high with volume confirmation → long
        # 2. 6h price breaks below 20-period Donchian low with volume confirmation → short
        # 3. Volatility regime filter: daily ATR ratio > 0.8 and < 2.0 (avoid extreme low/high vol)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: 6h breakout above Donchian high
        if (close[i] > highest_20[i] and            # 6h price above Donchian high
            volume_ratio[i] > 1.5 and               # Volume confirmation
            atr_ratio_6h[i] > 0.8 and               # Volatility not too low
            atr_ratio_6h[i] < 2.0):                 # Volatility not too high
            signals[i] = 0.25
            
        # Short conditions: 6h breakdown below Donchian low
        elif (close[i] < lowest_20[i] and           # 6h price below Donchian low
              volume_ratio[i] > 1.5 and             # Volume confirmation
              atr_ratio_6h[i] > 0.8 and             # Volatility not too low
              atr_ratio_6h[i] < 2.0):               # Volatility not too high
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_Volume_ATR_Regime_Filter"
timeframe = "6h"
leverage = 1.0