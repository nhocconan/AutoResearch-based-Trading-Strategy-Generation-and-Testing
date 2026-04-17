#!/usr/bin/env python3
"""
1h Donchian Breakout with 4h Trend Filter + Volume Spike
Long: Price > 20-period Donchian High + 4h EMA50 up + volume > 1.5x 20-bar avg
Short: Price < 20-period Donchian Low + 4h EMA50 down + volume > 1.5x 20-bar avg
Exit: Opposite Donchian breakout or volume drop below average
Uses 4h EMA for trend direction and 1h Donchian for precise entry timing.
Target: 60-150 total trades over 4 years (15-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 20-period Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume SMA for volume filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # need sufficient data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        ema_50_val = ema_50_4h_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + 4h EMA50 rising + volume spike
            if price > upper and ema_50_val > ema_50_4h_aligned[i-1] and vol > 1.5 * vol_sma_val:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Donchian low + 4h EMA50 falling + volume spike
            elif price < lower and ema_50_val < ema_50_4h_aligned[i-1] and vol > 1.5 * vol_sma_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian low or volume drops below average
            if price < lower or vol < vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price breaks above Donchian high or volume drops below average
            if price > upper or vol < vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian_Breakout_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0