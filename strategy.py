#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 15-period Donchian channels on 12h
    upper_15 = np.full_like(high_12h, np.nan)
    lower_15 = np.full_like(low_12h, np.nan)
    
    for i in range(len(high_12h)):
        if i < 14:
            upper_15[i] = np.nan
            lower_15[i] = np.nan
        else:
            upper_15[i] = np.max(high_12h[i-14:i+1])
            lower_15[i] = np.min(low_12h[i-14:i+1])
    
    # Align Donchian channels to 12h timeframe
    upper_15_aligned = align_htf_to_ltf(prices, df_12h, upper_15)
    lower_15_aligned = align_htf_to_ltf(prices, df_12h, lower_15)
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 10-period ATR on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align ATR to 12h timeframe
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # 20 for Donchian and volume
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_15_aligned[i]) or np.isnan(lower_15_aligned[i]) or
            np.isnan(atr_10_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian (15) with volume and volatility filter
            if price > upper_15_aligned[i] and vol > 1.5 * avg_vol[i] and atr_10_aligned[i] > 0:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian (15) with volume and volatility filter
            elif price < lower_15_aligned[i] and vol > 1.5 * avg_vol[i] and atr_10_aligned[i] > 0:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian
            if price < lower_15_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian
            if price > upper_15_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Volume_ATR_Filter"
timeframe = "12h"
leverage = 1.0