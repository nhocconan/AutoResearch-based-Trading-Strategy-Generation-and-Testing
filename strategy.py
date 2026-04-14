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
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly VWAP
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    vwap_1w = (typical_price_1w * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_1w = vwap_1w.values
    
    # Calculate 1d Williams %R (14-period)
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(vol_ma[i]):
            continue
        
        # Get 1d index for current 6h bar (6h = 0.25 * 1d)
        idx_1d = i // 4
        if idx_1d < 14:  # Need sufficient 1d data for Williams %R
            continue
            
        # Get 1w index for current 6h bar (6h = 1/28 * 1w approximately)
        idx_1w = i // 28
        if idx_1w < 14:  # Need sufficient 1w data for VWAP
            continue
        
        # Previous values to avoid look-ahead
        williams_prev = williams_r[idx_1d-1]
        vwap_prev = vwap_1w[idx_1w-1] if idx_1w-1 < len(vwap_1w) else vwap_1w[-1]
        
        if np.isnan(williams_prev):
            continue
        
        # Create arrays for alignment (constant values for the period)
        williams_arr = np.full(len(df_1d), williams_prev)
        vwap_arr = np.full(len(df_1w), vwap_prev)
        
        # Align to 6h timeframe
        williams_6h = align_htf_to_ltf(prices, df_1d, williams_arr)[i]
        vwap_6h = align_htf_to_ltf(prices, df_1w, vwap_arr)[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price above weekly VWAP + volume confirmation
            if (williams_6h < -80 and 
                close[i] > vwap_6h and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) + price below weekly VWAP + volume confirmation
            elif (williams_6h > -20 and 
                  close[i] < vwap_6h and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Williams %R returns to neutral (> -50) or price crosses below VWAP
            if williams_6h > -50 or close[i] < vwap_6h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Williams %R returns to neutral (< -50) or price crosses above VWAP
            if williams_6h < -50 or close[i] > vwap_6h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_1d_WilliamsR_1w_VWAP"
timeframe = "6h"
leverage = 1.0