#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d EMA Trend + Volume Spike
Long: Close > Donchian Upper(20), Close > 1d EMA50, Volume > 1.5x 4h Volume SMA(20)
Short: Close < Donchian Lower(20), Close < 1d EMA50, Volume > 1.5x 4h Volume SMA(20)
Exit: Opposite breakout or price crosses 1d EMA50
Designed for trend following in both bull and bear markets with trend filter.
Target: 80-150 total trades over 4 years (20-38/year)
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume SMA(20) for volume filter
    vol_sma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(40, 50)  # need Donchian(20), EMA50 and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_4h[i]
        ema_50_val = ema_50_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper, above 1d EMA50, volume spike
            if price > upper and price > ema_50_val and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower, below 1d EMA50, volume spike
            elif price < lower and price < ema_50_val and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian lower or crosses below 1d EMA50
            if price < lower or price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian upper or crosses above 1d EMA50
            if price > upper or price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0