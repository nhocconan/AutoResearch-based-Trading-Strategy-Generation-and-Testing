#!/usr/bin/env python3
"""
12h Donchian Breakout + 1w EMA Trend + Volume Spike
Long: Close breaks above Donchian(20) high + 1w EMA50 up + volume > 1.5x 12h volume SMA(20)
Short: Close breaks below Donchian(20) low + 1w EMA50 down + volume > 1.5x 12h volume SMA(20)
Exit: Opposite Donchian breakout or EMA trend change
Designed to capture strong trends with multi-timeframe confirmation and volume validation.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume SMA(20)
    vol_sma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(20, 50)  # need Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_sma_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_12h[i]
        ema_trend = ema_50_1w_aligned[i]
        ema_prev = ema_50_1w_aligned[i-1] if i > 0 else ema_trend
        
        if position == 0:
            # Long: break above Donchian high + EMA up + volume spike
            if price > highest_high[i] and ema_trend > ema_prev and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + EMA down + volume spike
            elif price < lowest_low[i] and ema_trend < ema_prev and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below Donchian low or EMA turns down
            if price < lowest_low[i] or ema_trend < ema_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above Donchian high or EMA turns up
            if price > highest_high[i] or ema_trend > ema_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0