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
    
    # Get daily data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) channel (based on previous 20 days)
    upper = np.full_like(high_1d, np.nan)
    lower = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(close_1d)):
        upper[i] = np.max(high_1d[i-20:i])
        lower[i] = np.min(low_1d[i-20:i])
    
    # Get 4h data for EMA filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # EMA(50) on 4h for trend filter
    ema_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all data to 4h timeframe
    upper_4h = align_htf_to_ltf(prices, df_1d, upper)
    lower_4h = align_htf_to_ltf(prices, df_1d, lower)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume, above EMA
            if close[i] > upper_4h[i] and vol_confirm and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume, below EMA
            elif close[i] < lower_4h[i] and vol_confirm and close[i] < ema_4h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR below EMA
            if close[i] < lower_4h[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR above EMA
            if close[i] > upper_4h[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA50_Volume_Filter"
timeframe = "4h"
leverage = 1.0