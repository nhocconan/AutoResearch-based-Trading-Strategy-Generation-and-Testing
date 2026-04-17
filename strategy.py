#!/usr/bin/env python3
"""
1h 4h/1d Trend + Volume Breakout with Session Filter
Long: Price breaks above 4h high + volume > 2x 4h volume SMA(20) + price > 1d EMA(50) + session (08-20 UTC)
Short: Price breaks below 4h low + volume > 2x 4h volume SMA(20) + price < 1d EMA(50) + session (08-20 UTC)
Exit: Reverse signal or session exit
Uses 4h for breakout levels, 1d for trend filter, 1h for entry timing
Target: 15-35 trades/year per symbol (60-140 total over 4 years)
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
    
    # Get 4h data for breakout levels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 4h high/low for breakout levels (previous 4h bar)
    high_4h_prev = np.roll(high_4h, 1)
    low_4h_prev = np.roll(low_4h, 1)
    high_4h_prev[0] = np.nan
    low_4h_prev[0] = np.nan
    
    # Align 4h levels to 1h
    high_4h_aligned = align_htf_to_ltf(prices, df_4h, high_4h_prev)
    low_4h_aligned = align_htf_to_ltf(prices, df_4h, low_4h_prev)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume SMA(20) for volume filter (use 4h volume from 4h data)
    vol_4h = df_4h['volume'].values
    vol_sma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_sma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_sma_4h)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 50)  # need EMA50 and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(high_4h_aligned[i]) or np.isnan(low_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_4h_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        high_4h_val = high_4h_aligned[i]
        low_4h_val = low_4h_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 4h high + volume > 2x SMA + price > 1d EMA50 + session
            if price > high_4h_val and close[i-1] <= high_4h_val and vol > 2.0 * vol_sma_val and price > ema_50_val:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h low + volume > 2x SMA + price < 1d EMA50 + session
            elif price < low_4h_val and close[i-1] >= low_4h_val and vol > 2.0 * vol_sma_val and price < ema_50_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Reverse signal or session exit
            if price < low_4h_val or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Reverse signal or session exit
            if price > high_4h_val or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_TrendVolumeBreakout_Session"
timeframe = "1h"
leverage = 1.0