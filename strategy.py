#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_volume_trend
Hypothesis: 4-hour strategy using Donchian channel breakout with daily trend filter and volume confirmation.
Breakouts above/below 4-hour Donchian(20) only when aligned with daily EMA(50) trend and volume > 1.5x average.
Designed to capture strong trends while minimizing false breakouts in choppy markets.
Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4-hour Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend
        uptrend = ema50_1d_aligned[i] > close_1d[0]  # Simplified: compare to first available value
        # Better approach: use slope of EMA
        if i >= 21:
            ema50_prev = ema50_1d_aligned[i-1]
            ema50_curr = ema50_1d_aligned[i]
            uptrend = ema50_curr > ema50_prev
            downtrend = ema50_curr < ema50_prev
        else:
            uptrend = ema50_1d_aligned[i] > ema50_1d_aligned[max(0, i-1)]
            downtrend = ema50_1d_aligned[i] < ema50_1d_aligned[max(0, i-1)]
        
        # Long breakout: price breaks above Donchian high with uptrend and volume
        if (close[i] > high_20[i]) and uptrend and volume_ok[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short breakdown: price breaks below Donchian low with downtrend and volume
        elif (close[i] < low_20[i]) and downtrend and volume_ok[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit on opposite breakout
        elif position == 1 and (close[i] < low_20[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > high_20[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_trend"
timeframe = "4h"
leverage = 1.0