#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation
- Uses Donchian channel (20-period high/low) from 4h timeframe for breakout entries
- 1d EMA50 defines higher timeframe trend filter: only trade in direction of 1d trend
- Volume confirmation (> 1.8x 20-period average) filters weak breakouts
- Exit when price retouches the midpoint of the Donchian channel or trend reverses
- Designed for 4h timeframe targeting 20-35 trades/year (80-140 over 4 years)
- Works in both bull and bear markets by combining breakout momentum with trend filter
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channel (20-period) on 4h data
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_ma + low_ma) / 2
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND above 1d EMA50 AND volume spike
            if (close[i] > high_ma[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND below 1d EMA50 AND volume spike
            elif (close[i] < low_ma[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price retouches Donchian midpoint OR trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when price retouches midpoint OR closes below 1d EMA50
                if (close[i] <= donchian_mid[i] or close[i] < ema_50_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when price retouches midpoint OR closes above 1d EMA50
                if (close[i] >= donchian_mid[i] or close[i] > ema_50_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0