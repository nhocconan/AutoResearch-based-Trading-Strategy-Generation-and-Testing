#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend and structure
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d data for volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper and lower bands
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR for volatility filter
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - np.concatenate([[high_4h[0]], high_4h[:-1]]))
    tr3 = np.abs(low_4h[1:] - np.concatenate([[low_4h[0]], low_4h[:-1]]))
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d volume average for context
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 1h timeframe
    donchian_upper_1h = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_1h = align_htf_to_ltf(prices, df_4h, donchian_lower)
    atr_4h_1h = align_htf_to_ltf(prices, df_4h, atr_4h)
    volume_ma_20_1d_1h = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Skip if not in session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not ready
        if (np.isnan(donchian_upper_1h[i]) or np.isnan(donchian_lower_1h[i]) or
            np.isnan(atr_4h_1h[i]) or np.isnan(volume_ma_20_1d_1h[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x 1d average (adjusted for timeframe)
        volume_condition = volume[i] > (volume_ma_20_1d_1h[i] * 1.3)
        
        # ATR filter: avoid extremely low volatility
        atr_condition = atr_4h_1h[i] > 0
        
        if position == 0:
            # Long: break above Donchian upper with volume and volatility
            if close[i] > donchian_upper_1h[i] and volume_condition and atr_condition:
                position = 1
                signals[i] = position_size
            # Short: break below Donchian lower with volume and volatility
            elif close[i] < donchian_lower_1h[i] and volume_condition and atr_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches or crosses below Donchian lower
            if close[i] < donchian_lower_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price touches or crosses above Donchian upper
            if close[i] > donchian_upper_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_Donchian_Breakout_Volume_Filter"
timeframe = "1h"
leverage = 1.0