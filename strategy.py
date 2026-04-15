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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align 1w Donchian to 1d
    upper_20_1d = align_htf_to_ltf(prices, df_1w, upper_20_1w)
    lower_20_1d = align_htf_to_ltf(prices, df_1w, lower_20_1w)
    
    # Get 1d data for volume and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_1d / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: UTC 00-24 (always true for 1d, kept for structure)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_1d[i]) or np.isnan(lower_20_1d[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 1d price breaks above 1w Donchian upper (20) - bullish breakout
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close_1d[i] > upper_20_1d[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close_1d[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 1d price breaks below 1w Donchian lower (20) - bearish breakdown
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.5% of price
        elif (close_1d[i] < lower_20_1d[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close_1d[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_1w_Donchian20_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0