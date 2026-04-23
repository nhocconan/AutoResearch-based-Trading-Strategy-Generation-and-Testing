#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) Breakout with 12h EMA50 Trend Filter and Volume Spike
- Uses Donchian(20) from 4h for breakout signals
- 12h EMA50 defines medium-term trend: only long when price > EMA50, short when price < EMA50
- Volume confirmation (> 2.0x 20-period average) filters weak breakouts
- Designed for 4h timeframe targeting 20-40 trades/year (80-160 over 4 years)
- Works in both bull and bear markets by following the 12h EMA50 trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need 12h EMA50, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian(20) on 4h data up to current bar
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            # Not enough data for Donchian yet
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND above 12h EMA50 AND volume spike
            if (close[i] > donchian_high and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 12h EMA50 AND volume spike
            elif (close[i] < donchian_low and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Donchian midpoint OR crosses 12h EMA50
            exit_signal = False
            donchian_mid = (donchian_high + donchian_low) / 2
            
            if position == 1:
                # Exit long when price < Donchian midpoint OR < 12h EMA50
                if close[i] < donchian_mid or close[i] < ema_50_12h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > Donchian midpoint OR > 12h EMA50
                if close[i] > donchian_mid or close[i] > ema_50_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0