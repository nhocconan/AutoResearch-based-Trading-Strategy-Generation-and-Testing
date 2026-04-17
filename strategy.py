#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + Volume Spike + 1w Trend Filter
Long: Price breaks above 20-period Donchian high + volume > 2x 12h volume SMA(20) + price > 1w EMA(50)
Short: Price breaks below 20-period Donchian low + volume > 2x 12h volume SMA(20) + price < 1w EMA(50)
Exit: Price retests the opposite Donchian level (high for short, low for long) or ATR-based stop
Uses Donchian channels for breakout, volume confirmation, and weekly trend filter
Target: 15-25 trades/year per symbol (60-100 total over 4 years)
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h volume SMA(20) for volume filter
    vol_sma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(40, 50)  # need EMA50 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_sma_12h[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_12h[i]
        ema_50_val = ema_50_1w_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume > 2x SMA + price > 1w EMA50
            if price > upper and high[i-1] <= upper and vol > 2.0 * vol_sma_val and price > ema_50_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume > 2x SMA + price < 1w EMA50
            elif price < lower and low[i-1] >= lower and vol > 2.0 * vol_sma_val and price < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price retests Donchian low or breaks below it
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price retests Donchian high or breaks above it
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_VolumeSpike_1wEMA50"
timeframe = "12h"
leverage = 1.0