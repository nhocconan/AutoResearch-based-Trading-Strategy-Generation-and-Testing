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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 4h
    upper_20_4h = align_htf_to_ltf(prices, df_1d, upper_20_1d)
    lower_20_4h = align_htf_to_ltf(prices, df_1d, lower_20_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[low_1d[0]], low_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 4h - less restrictive)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 4h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(atr_14_4h[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above 1d Donchian upper (20) - bullish breakout
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > upper_20_4h[i] and
            volume_ratio[i] > 1.5 and
            atr_14_4h[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 1d Donchian lower (20) - bearish breakdown
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.5% of price
        elif (close[i] < lower_20_4h[i] and
              volume_ratio[i] > 1.5 and
              atr_14_4h[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Donchian20_Volume_ATR_Filter_v2"
timeframe = "4h"
leverage = 1.0