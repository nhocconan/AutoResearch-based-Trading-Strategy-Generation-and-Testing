# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Strategy: 1h EMA Pullback with 4h Trend and 1d Volume Confirmation
Hypothesis: In trending markets (identified by 4h EMA alignment), pullbacks to the 1h EMA offer high-probability entries. 
Volume confirmation from 1d average volume filters low-conviction moves. Works in bull via trend continuation, 
in bear via short entries on rallies to resistance. Target: 15-35 trades/year.
"""
name = "1h_ema_pullback_4h1d_volume_v2"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20, center=False).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1h EMA(20) for pullback entries
    ema_1h = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_1h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1d average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA(20) OR trend changes
            if close[i] < ema_1h[i] or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above EMA(20) OR trend changes
            if close[i] > ema_1h[i] or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price pulls back to EMA(20) from above in uptrend + volume confirmation
            if (close[i] >= ema_1h[i] and  # Price at or above EMA
                close[i-1] < ema_1h[i-1] and  # Was below EMA (pullback completion)
                close[i] > ema_4h_aligned[i] and  # Uptrend: price above 4h EMA
                vol_confirm):
                position = 1
                signals[i] = 0.20
            # Enter short: price pulls back to EMA(20) from below in downtrend + volume confirmation
            elif (close[i] <= ema_1h[i] and  # Price at or below EMA
                  close[i-1] > ema_1h[i-1] and  # Was above EMA (pullback completion)
                  close[i] < ema_4h_aligned[i] and  # Downtrend: price below 4h EMA
                  vol_confirm):
                position = -1
                signals[i] = -0.20
    
    return signals