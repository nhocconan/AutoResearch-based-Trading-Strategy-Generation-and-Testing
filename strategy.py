#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) Breakout + 1w EMA50 Trend Filter + Volume Spike
- Uses 20-period Donchian channel from 12h timeframe for structure-based breakouts
- 1w EMA50 defines weekly trend filter: only trade in direction of weekly trend
- Volume confirmation (> 1.8x 20-period average) filters weak signals
- Exit when price retouches Donchian midpoint or trend reverses
- Designed for 12h timeframe targeting 15-25 trades/year (60-100 over 4 years)
- Works in bull markets via breakouts, in bear markets via trend-filtered shorts
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
    
    # Calculate weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Donchian(20) channels
    # Use rolling window on 12h data directly
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = high_roll
    lower_channel = low_roll
    mid_channel = (upper_channel + lower_channel) / 2
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(mid_channel[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper channel AND above weekly EMA50 AND volume spike
            if (close[i] > upper_channel[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel AND below weekly EMA50 AND volume spike
            elif (close[i] < lower_channel[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price retouches midpoint OR trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when price retouches midpoint OR closes below weekly EMA50
                if (close[i] <= mid_channel[i] or close[i] < ema_50_1w_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when price retouches midpoint OR closes above weekly EMA50
                if (close[i] >= mid_channel[i] or close[i] > ema_50_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0