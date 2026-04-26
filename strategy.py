#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
Only go long when price breaks above 20-period 12h high AND price > 1d EMA50 AND volume > 1.5x 20-bar MA.
Only go short when price breaks below 20-period 12h low AND price < 1d EMA50 AND volume > 1.5x 20-bar MA.
Exit on opposite Donchian breakout or trend reversal.
Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year per symbol.
Works in bull/bear via trend filter - only long in uptrend, short in downtrend.
Volume filter prevents low-conviction entries.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detector (20-bar volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-period high, uptrend, volume spike
            if close[i] > high_ma[i] and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low, downtrend, volume spike
            elif close[i] < low_ma[i] and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below 20-period low OR trend changes to downtrend
            if close[i] < low_ma[i] or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above 20-period high OR trend changes to uptrend
            if close[i] > high_ma[i] or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0