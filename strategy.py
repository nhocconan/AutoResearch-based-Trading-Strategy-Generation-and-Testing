#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend + volume spike.
Long when price breaks above 20-period high with volume > 1.5x average and close > 1d EMA34.
Short when price breaks below 20-period low with volume > 1.5x average and close < 1d EMA34.
Exit when price reverts to 20-period midpoint or opposite breakout occurs.
Uses 1d EMA for trend filter, 12h for price/volume/Donchian.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    midpoint = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i-donchian_period+1:i+1])
        lower[i] = np.min(low[i-donchian_period+1:i+1])
        midpoint[i] = (upper[i] + lower[i]) / 2.0
    
    # Calculate volume spike (current volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, donchian_period)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(midpoint[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema_trend = ema_34_1d_aligned[i]
        up = upper[i]
        low_ch = lower[i]
        mid = midpoint[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike and bullish trend
            if price > up and vol_spike and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume spike and bearish trend
            elif price < low_ch and vol_spike and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint OR breaks below lower (opposite signal)
            if price <= mid or price < low_ch:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint OR breaks above upper (opposite signal)
            if price >= mid or price > up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0