#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) Breakout with 1d EMA34 Trend Filter and Volume Spike
- Uses Donchian(20) from 12h for breakout signals
- 1d EMA34 defines long-term trend: only long when price > EMA34, short when price < EMA34
- Volume confirmation (> 2.0x 20-period average) filters weak breakouts
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by following the 1d EMA34 trend filter
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need 1d EMA34, Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian(20) on 12h data up to current bar
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
            # Long: price breaks above Donchian high AND above 1d EMA34 AND volume spike
            if (close[i] > donchian_high and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 1d EMA34 AND volume spike
            elif (close[i] < donchian_low and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Donchian midpoint OR crosses 1d EMA34
            exit_signal = False
            donchian_mid = (donchian_high + donchian_low) / 2
            
            if position == 1:
                # Exit long when price < Donchian midpoint OR < 1d EMA34
                if close[i] < donchian_mid or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > Donchian midpoint OR > 1d EMA34
                if close[i] > donchian_mid or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0