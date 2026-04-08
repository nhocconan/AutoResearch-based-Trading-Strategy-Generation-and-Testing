#!/usr/bin/env python3
"""
1h Volume Breakout with 4h/1d Trend Filter
Hypothesis: Volume spikes on 1h capture short-term momentum bursts, filtered by 4h and 1d EMA trends.
Only trade in the direction of both higher timeframe trends to reduce whipsaw.
Session filter (08-20 UTC) removes low-liquidity periods. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_breakout_4h1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_4h = df_4h['close'].ewm(span=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_1d = df_1d['close'].ewm(span=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: >2x 24-period average on 1h
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_filter[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below either EMA or volume dries up
            if close[i] < ema_4h_aligned[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above either EMA or volume dries up
            if close[i] > ema_4h_aligned[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Breakout long with both trends aligned up
            if (close[i] > ema_4h_aligned[i] and 
                close[i] > ema_1d_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.20
            # Breakout short with both trends aligned down
            elif (close[i] < ema_4h_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.20
    
    return signals