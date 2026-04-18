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
    
    # Get daily data for 20-period Donchian (structure) and 200-period EMA (trend)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian upper/lower on daily
    high_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA200 on daily
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align to 4h timeframe
    donchian_up = align_htf_to_ltf(prices, df_1d, high_20d)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_20d)
    ema200_4h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # need Donchian and EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_up[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema200_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        vol_confirmed = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price above Donchian upper, above EMA200, with volume
            if (close[i] > donchian_up[i] and 
                close[i] > ema200_4h[i] and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price below Donchian lower, below EMA200, with volume
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema200_4h[i] and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below EMA200 or below Donchian lower
            if close[i] < ema200_4h[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA200 or above Donchian upper
            if close[i] > ema200_4h[i] or close[i] > donchian_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_EMA200_Volume_Breakout"
timeframe = "4h"
leverage = 1.0