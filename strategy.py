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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period) - price channel structure
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 4h
    upper_20_4h = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_4h = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate 1d ATR(14) for volatility filter
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift())
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr_1d = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    volume_ratio = vol_4h / (vol_ma_20_4h_aligned + 1e-10)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(volume_ratio[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above 1d Donchian upper (20)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        if (close[i] > upper_20_4h[i] and
            volume_ratio[i] > 1.5 and
            atr_14_1d_aligned[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 1d Donchian lower (20)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.3% of price
        elif (close[i] < lower_20_4h[i] and
              volume_ratio[i] > 1.5 and
              atr_14_1d_aligned[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Donchian20_Volume_ATR_Filter_v1"
timeframe = "4h"
leverage = 1.0