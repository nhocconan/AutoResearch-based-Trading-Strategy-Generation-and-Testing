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
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily ATR (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily Donchian channel (20)
    upper_dc_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_dc_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Get aligned daily indicators
        atr_14_1d_i = align_htf_to_ltf(prices, df_1d, atr_14_1d)[i]
        upper_dc_20_i = align_htf_to_ltf(prices, df_1d, upper_dc_20)[i]
        lower_dc_20_i = align_htf_to_ltf(prices, df_1d, lower_dc_20)[i]
        
        if np.isnan(atr_14_1d_i) or np.isnan(upper_dc_20_i) or np.isnan(lower_dc_20_i):
            continue
        
        # Volatility filter: ATR below 50-period average (low vol regime)
        atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
        atr_ma_50_1d_i = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)[i]
        if np.isnan(atr_ma_50_1d_i):
            continue
        low_vol = atr_14_1d_i < 0.7 * atr_ma_50_1d_i  # Below 70% of MA
        
        # Volume spike filter (1.5x median volume)
        vol_median = np.nanmedian(volume[:i+1])  # Use historical median
        volume_spike = volume[i] > 1.5 * vol_median
        
        # Long: break above upper Donchian with volume in low vol
        if position == 0 and low_vol and volume_spike:
            if close[i] > upper_dc_20_i:
                position = 1
                signals[i] = position_size
            # Short: break below lower Donchian with volume in low vol
            elif close[i] < lower_dc_20_i:
                position = -1
                signals[i] = -position_size
        
        # Exit: price returns to midpoint of Donchian channel
        elif position != 0:
            mid_dc = (upper_dc_20_i + lower_dc_20_i) / 2
            if position == 1 and close[i] < mid_dc:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > mid_dc:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_LowVol_Volume"
timeframe = "12h"
leverage = 1.0