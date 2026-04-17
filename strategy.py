#!/usr/bin/env python3
"""
12h Donchian Breakout Volume Spike + 1d Trend Filter
Long: Close > Donchian Upper(20) + Volume > 2x 12h Volume SMA(20) + Price > 1d EMA50
Short: Close < Donchian Lower(20) + Volume > 2x 12h Volume SMA(20) + Price < 1d EMA50
Exit: Opposite breakout or price crosses 1d EMA50
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume SMA(20) for volume filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 50)  # need EMA50 and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i]) or
            np.isnan(high_max_20[i]) or np.isnan(low_min_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        ema_50_val = ema_50_1d_aligned[i]
        upper = high_max_20[i]
        lower = low_min_20[i]
        
        if position == 0:
            # Long: Close > Donchian Upper + Volume Spike + Price > 1d EMA50
            if price > upper and vol > 2.0 * vol_sma_val and price > ema_50_val:
                signals[i] = 0.25
                position = 1
            # Short: Close < Donchian Lower + Volume Spike + Price < 1d EMA50
            elif price < lower and vol > 2.0 * vol_sma_val and price < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Opposite breakout or price crosses below 1d EMA50
            if price < lower or price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Opposite breakout or price crosses above 1d EMA50
            if price > upper or price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_VolumeSpike_1dEMA50"
timeframe = "12h"
leverage = 1.0