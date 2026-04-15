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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align 1w Donchian to 12h
    upper_20_12h = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_12h = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # Get 1d HTF data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 12h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume ratio (current vs 1d 20-period average)
    vol_ma_20_1d_safe = vol_ma_20_1d_aligned + 1e-10
    volume_ratio = volume / vol_ma_20_1d_safe
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_12h[i]) or np.isnan(lower_20_12h[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 12h price breaks above 1w Donchian upper (20)
        # 2. Volume confirmation: volume > 2.0x 1d average
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > upper_20_12h[i] and
            volume_ratio[i] > 2.0 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 12h price breaks below 1w Donchian lower (20)
        # 2. Volume confirmation: volume > 2.0x 1d average
        # 3. Volatility filter: ATR > 0.5% of price
        elif (close[i] < lower_20_12h[i] and
              volume_ratio[i] > 2.0 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1w_Donchian20_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0