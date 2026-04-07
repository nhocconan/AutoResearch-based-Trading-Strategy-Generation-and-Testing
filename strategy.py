#!/usr/bin/env python3
"""
1h_volume_breakout_4h1d_trend_v1
Hypothesis: On 1-hour timeframe, enter long when price breaks above 24-period high with volume > 2x 24-period average and 4h/1d trend up; short when price breaks below 24-period low with volume > 2x average and 4h/1d trend down. Exit when price returns to 24-period midpoint. Uses 4h/1d for trend direction and 1h for timing to avoid counter-trend trades. Designed for 15-30 trades/year to minimize fee drag while capturing momentum in both bull and bear markets.
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
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Determine trend direction (using EMA slope)
    trend_4h_up = np.zeros(len(ema_20_4h_aligned), dtype=bool)
    trend_4h_down = np.zeros(len(ema_20_4h_aligned), dtype=bool)
    trend_1d_up = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    trend_1d_down = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    
    for i in range(1, len(ema_20_4h_aligned)):
        if not np.isnan(ema_20_4h_aligned[i]) and not np.isnan(ema_20_4h_aligned[i-1]):
            trend_4h_up[i] = ema_20_4h_aligned[i] > ema_20_4h_aligned[i-1]
            trend_4h_down[i] = ema_20_4h_aligned[i] < ema_20_4h_aligned[i-1]
    
    for i in range(1, len(ema_50_1d_aligned)):
        if not np.isnan(ema_50_1d_aligned[i]) and not np.isnan(ema_50_1d_aligned[i-1]):
            trend_1d_up[i] = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            trend_1d_down[i] = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
    
    # Calculate 24-period high/low and midpoint on 1h
    period = 24
    high_max = pd.Series(high).rolling(window=period, min_periods=period).max().values
    low_min = pd.Series(low).rolling(window=period, min_periods=period).min().values
    midpoint = (high_max + low_min) / 2
    
    # Volume filter: 24-period average
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(period, 50), n):
        # Skip if data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: at least 2x average
        vol_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to midpoint
            if close[i] <= midpoint[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price returns to midpoint
            if close[i] >= midpoint[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Only enter with volume confirmation and trend alignment on both 4h and 1d
            if vol_ok:
                # Long: price breaks above 24-period high with 4h and 1d uptrend
                if (close[i] > high_max[i] and close[i-1] <= high_max[i-1] and 
                    trend_4h_up[i] and trend_1d_up[i]):
                    position = 1
                    signals[i] = 0.20
                # Short: price breaks below 24-period low with 4h and 1d downtrend
                elif (close[i] < low_min[i] and close[i-1] >= low_min[i-1] and 
                      trend_4h_down[i] and trend_1d_down[i]):
                    position = -1
                    signals[i] = -0.20
    
    return signals