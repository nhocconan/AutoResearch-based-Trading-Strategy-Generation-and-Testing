#!/usr/bin/env python3
"""
1h Volume Spike + 4h/1d EMA Trend Filter with Time-of-Day Filter
Long: Volume > 2x 20-period SMA AND price > 4h EMA(20) AND price > 1d EMA(50)
Short: Volume > 2x 20-period SMA AND price < 4h EMA(20) AND price < 1d EMA(50)
Exit: When volume condition fails OR trend reverses
Time filter: Only trade between 08:00-20:00 UTC
Target: 20-30 trades/year per symbol (80-120 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA(20) for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume SMA(20) for volume filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute hour for session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(20, 50)  # need EMA50 and volume SMA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        ema_20_4h_val = ema_20_4h_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: High volume + price above both EMAs
            if vol > 2.0 * vol_sma_val and price > ema_20_4h_val and price > ema_50_1d_val:
                signals[i] = 0.20
                position = 1
            # Short: High volume + price below both EMAs
            elif vol > 2.0 * vol_sma_val and price < ema_20_4h_val and price < ema_50_1d_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Volume drops OR price breaks below either EMA
            if vol <= 2.0 * vol_sma_val or price < ema_20_4h_val or price < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Volume drops OR price breaks above either EMA
            if vol <= 2.0 * vol_sma_val or price > ema_20_4h_val or price > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_4hEMA20_1dEMA50_TimeFilter"
timeframe = "1h"
leverage = 1.0